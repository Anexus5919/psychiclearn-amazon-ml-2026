"""Run the pipeline as a self-terminating EC2 batch job (Free-plan friendly; no endpoints).

Subcommands
  setup  : create (idempotent) the instance role/profile, scoped to one S3 bucket only
  launch : package src/ + requirements, upload to S3, start an instance that runs the pipeline,
           uploads logs/outputs to s3://<bucket>/runs/<run-id>/ and then TERMINATES itself
           (a hard shutdown timer caps the run length even if the job hangs)
  status : print instance state and the tail of the job log from S3
  fetch  : download the run's outputs and reports

Data is expected at s3://<bucket>/<data-prefix>/{train,test}/*.tsv[.gz] (default data/raw).
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time

import boto3
from botocore.exceptions import ClientError

ROLE = "PsychicLearnBatchEC2Role"
UBUNTU_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # business_entity_resolution/

BOOTSTRAP = r"""#!/bin/bash
# Batch-job bootstrap (Ubuntu 24.04, root). Always ends by powering off -> instance terminates.
set -o pipefail
BUCKET=__BUCKET__; RUN=__RUN__; REUSE=__REUSE__; DATA=__DATA__
exec > /var/log/ber-job.log 2>&1
shutdown -h +__MAXMIN__
export HOME=/root DEBIAN_FRONTEND=noninteractive
snap install aws-cli --classic || { apt-get update -y && apt-get install -y awscli; }
AWS=$(command -v aws || echo /snap/bin/aws)
up() { $AWS s3 cp "$1" "s3://$BUCKET/runs/$RUN/$2" --only-show-errors; }
sync_logs() { up /var/log/ber-job.log job.log; [ -f /work/log.txt ] && up /work/log.txt pipeline.log; true; }
( while true; do sync_logs; sleep 60; done ) &
mkdir -p /opt/ber /data /work /out
$AWS s3 cp "s3://$BUCKET/runs/$RUN/code.tar.gz" /opt/ber/code.tar.gz --only-show-errors && tar -xzf /opt/ber/code.tar.gz -C /opt/ber
$AWS s3 cp "s3://$BUCKET/$DATA/" /data/ --recursive --exclude "MD5*" --only-show-errors
if [ -n "$REUSE" ]; then $AWS s3 cp "s3://$BUCKET/runs/$REUSE/work/" /work/ --recursive --only-show-errors --exclude "log.txt" --exclude "report_*"; fi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv venv /opt/venv --python 3.12 && uv pip install --python /opt/venv/bin/python -r /opt/ber/requirements.txt
echo "=== nproc $(nproc)  mem $(free -g | awk '/Mem/{print $2}')G"
cd /opt/ber/src && /opt/venv/bin/python -m ber.pipeline --data-dir /data --work-dir /work --out-dir /out __ARGS__
echo "PIPELINE_EXIT=$?"
if [ -f /out/matching_results.tsv ]; then
  for f in /data/test/*.gz; do gunzip -kf "$f"; done
  /opt/venv/bin/python /opt/ber/validate_submission.py --matching /out/matching_results.tsv \
      --candidate /out/candidate_pairs.tsv --test-dir /data/test --check-ids > /out/validator.txt 2>&1
  echo "VALIDATOR_EXIT=$?"; cat /out/validator.txt
  gzip -kf /out/candidate_pairs.tsv
  for f in matching_results.tsv candidate_pairs.tsv.gz validator.txt; do up /out/$f out/$f; done
fi
for f in /work/report_train.json /work/report_predict.json; do [ -f $f ] && up $f $(basename $f); done
$AWS s3 cp /work/ "s3://$BUCKET/runs/$RUN/work/" --recursive --only-show-errors --exclude "*.tsv"
sync_logs; echo DONE > /tmp/DONE; up /tmp/DONE DONE
shutdown -h now
"""


def session(profile, region):
    return boto3.Session(profile_name=profile, region_name=region)


def ensure_instance_profile(sess, bucket):
    """Create the EC2 role (trust: ec2.amazonaws.com) with S3 access to ONE bucket, and its profile."""
    iam = sess.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "ListBucket", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}"},
        {"Sid": "ReadWriteObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": f"arn:aws:s3:::{bucket}/*"}]}
    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="PsychicLearn batch job - S3 access to one bucket only")
        print(f"created role {ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"role {ROLE} exists")
    iam.put_role_policy(RoleName=ROLE, PolicyName="S3SingleBucket", PolicyDocument=json.dumps(policy))
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE)
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE, RoleName=ROLE)
        iam.get_waiter("instance_profile_exists").wait(InstanceProfileName=ROLE)
        time.sleep(10)  # instance profiles need a few seconds before EC2 can use them
        print(f"created instance profile {ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"instance profile {ROLE} exists")


def package_code():
    """tar.gz of src/, requirements.txt and the official validator (if present next to the package)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(PKG_ROOT, "src"), arcname="src",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add(os.path.join(PKG_ROOT, "requirements.txt"), arcname="requirements.txt")
        validator = os.environ.get("BER_VALIDATOR")
        if validator and os.path.exists(validator):
            tar.add(validator, arcname="validate_submission.py")
    return buf.getvalue()


def launch(sess, bucket, run_id, instance_type, max_hours, pipeline_args, reuse, volume_gb, data_prefix):
    s3 = sess.client("s3")
    s3.put_object(Bucket=bucket, Key=f"runs/{run_id}/code.tar.gz", Body=package_code())
    ami = sess.client("ssm").get_parameter(Name=UBUNTU_PARAM)["Parameter"]["Value"]
    user_data = (BOOTSTRAP.replace("__BUCKET__", bucket).replace("__RUN__", run_id)
                 .replace("__REUSE__", reuse or "").replace("__DATA__", data_prefix).replace("__MAXMIN__", str(int(max_hours * 60)))
                 .replace("__ARGS__", pipeline_args))
    ec2 = sess.client("ec2")
    resp = ec2.run_instances(
        ImageId=ami, InstanceType=instance_type, MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": ROLE}, UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"psychiclearn-{run_id}"}, {"Key": "project", "Value": "psychiclearn"}]}],
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({instance_type}, ami {ami}); run s3://{bucket}/runs/{run_id}/ ; hard cap {max_hours} h")
    return iid


def status(sess, bucket, run_id, tail):
    ec2, s3 = sess.client("ec2"), sess.client("s3")
    res = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [f"psychiclearn-{run_id}"]}])
    for r in res["Reservations"]:
        for i in r["Instances"]:
            print(f"instance {i['InstanceId']} {i['InstanceType']} state={i['State']['Name']} launched={i['LaunchTime']}")
    for key in ("pipeline.log", "job.log"):
        try:
            with s3.get_object(Bucket=bucket, Key=f"runs/{run_id}/{key}")["Body"] as body:
                lines = body.read().decode("utf-8", "replace").splitlines()
            print(f"--- {key} (last {tail} lines)")
            print("\n".join(lines[-tail:]))
        except s3.exceptions.NoSuchKey:
            print(f"--- {key}: not yet available")
    try:
        s3.head_object(Bucket=bucket, Key=f"runs/{run_id}/DONE")
        print("RUN DONE")
    except ClientError:
        pass


def fetch(sess, bucket, run_id, dest):
    s3 = sess.client("s3")
    os.makedirs(dest, exist_ok=True)
    for key in ("out/matching_results.tsv", "out/candidate_pairs.tsv.gz", "out/validator.txt",
                "report_train.json", "report_predict.json", "pipeline.log"):
        path = os.path.join(dest, os.path.basename(key))
        try:
            s3.download_file(bucket, f"runs/{run_id}/{key}", path)
            print(f"downloaded {key}")
        except ClientError as e:
            print(f"skip {key}: {e.response['Error']['Code']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "launch", "status", "fetch"])
    ap.add_argument("--profile", default="psychiclearn1")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--instance-type", default="r7a.xlarge")
    ap.add_argument("--max-hours", type=float, default=10.0, help="hard cap; the instance powers off (terminates) after this")
    ap.add_argument("--volume-gb", type=int, default=80)
    ap.add_argument("--data-prefix", default="data/raw", help="S3 prefix holding train/ and test/")
    ap.add_argument("--reuse", default="", help="run id whose work/ directory is restored first (skips finished stages)")
    ap.add_argument("--pipeline-args", default="", help="extra arguments for ber.pipeline, e.g. '--train-frac 0.15'")
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--dest", default="runs_fetched")
    a = ap.parse_args()
    sess = session(a.profile, a.region)
    try:
        if a.command == "setup":
            ensure_instance_profile(sess, a.bucket)
        elif a.command == "launch":
            launch(sess, a.bucket, a.run_id, a.instance_type, a.max_hours, a.pipeline_args, a.reuse, a.volume_gb, a.data_prefix)
        elif a.command == "status":
            status(sess, a.bucket, a.run_id, a.tail)
        else:
            fetch(sess, a.bucket, a.run_id, os.path.join(a.dest, a.run_id))
        return 0
    except ClientError as e:
        print(f"AWS error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
