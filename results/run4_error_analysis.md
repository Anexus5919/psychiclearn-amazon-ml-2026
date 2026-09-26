# Run-4 error analysis (26 Sep 2026, ~16:30 IST)

Data: run-4 validation (out-of-fold) predictions for 304,555 training businesses (India 121,885, US 182,670), with thresholds t1 0.54 / t2 0.74. Scripts: `dev_run5/scripts/err4.py`, `miss4.py`.

## 1. Error budget

| | India | US |
|---|---|---|
| Validation F0.5 | 0.96710 | 0.98240 |
| True pairs | 422,542 | 632,310 |
| Never retrieved | 23,602 (5.59%) | 11,004 (1.74%) |
| Dropped by pruning | 1,133 (0.27%) | 463 (0.07%) |
| F0.5 if all false merges removed | 0.97064 (+0.0035) | 0.98543 (+0.0030) |
| F0.5 if all retrieved-but-rejected true matches accepted | 0.97349 (+0.0064) | 0.99107 (+0.0087) |
| Rejected true matches (model false negatives) | 8,271 (5,173 had an e5 score; 3,768 with e5 > 0.5) | 16,996 (11,784 scored; 6,802 with e5 > 0.5) |

False merges all have p > 0.54 (the threshold): they are confident errors.

## 2. Why true matches are never retrieved (missed vs found pairs, after normalisation)

| | India missed | India found | US missed | US found |
|---|---|---|---|---|
| Name similarity ≥ 90 (token-set) | **75.8%** | 84.4% | **57.9%** | 82.9% |
| Name similarity < 60 | 10.0% | 5.2% | 14.9% | 4.5% |
| Address similarity ≥ 80 | 62.1% | 94.4% | 29.5% | 89.3% |
| Candidate address empty | 19.2% | 3.1% | 45.8% | 4.2% |
| Candidate name in an Indic script | 34.0% | 17.3% | 0% | 0% |

→ The main cause is **crowding**: same-named businesses elsewhere in the country fill the top-k name list. Fix in run 5: a state/region-restricted name search.

## 3. Name crowding per country (test S2 records sharing their exact name with > 15 others)

| | Country-wide | Within state/region | Within city (estimate from a 200k sample) |
|---|---|---|---|
| France | 19.5% | 12.7% | ~9.8% |
| India | 24.3% | 7.9% | ~5.7% |
| US | 9.5% | 0.7% | ~2.0% |

## 4. Region key quality (run 5, held-out true pairs of query businesses)

| | Both regions found | Same region (when both found) |
|---|---|---|
| India | 84.5% (33,897 / 40,101) | **98.4%** |
| US | 94.9% (56,827 / 59,899) | **93.1%** |

Regions learned: India 15 states, US 45 states, France 3 regions. Coverage in the test pools: India ~86%, US ~96–97%, France ~96%.

## 5. Candidate-file size vs recall (run-4 validation)

| Keep pairs with matcher p ≥ | Candidates per business | Candidate recall |
|---|---|---|
| (all, run 4) | 22.6 | 0.9657 |
| 0.0001 (used from run 5) | 7.3 | 0.9657 (−0.0027%) |
| 0.0005 | 5.3 | 0.9655 (−0.013%) |
| 0.001 | 4.9 | 0.9654 (−0.025%) |
| 0.01 | 3.9 | 0.9640 (−0.17%) |

## 6. Raw output of err4.py (with examples)

```
raw retrieved pairs: 19242452

== India: n=121,885 true pairs=422,542  F0.5=0.96710
   retrieval misses 23,602 (5.59%)   pruning misses 1,133 (0.27%)
   remove all FP -> 0.97064 (+0.0035)   add all model-missed (in cands) -> 0.97349 (+0.0064)
   model FN by p: {Interval(0.0, 0.02, closed='right'): 766, Interval(0.02, 0.1, closed='right'): 1026, Interval(0.1, 0.3, closed='right'): 1770, Interval(0.3, 0.54, closed='right'): 2299, Interval(0.54, 0.74, closed='right'): 2392, Interval(0.74, 1.01, closed='right'): 18}
   model FN with ce_p (in band): 5173 of 8271 | FN ce_p>0.5: 3768
   FP by p: {Interval(0.0, 0.02, closed='right'): 0, Interval(0.02, 0.1, closed='right'): 0, Interval(0.1, 0.3, closed='right'): 0, Interval(0.3, 0.54, closed='right'): 0, Interval(0.54, 0.74, closed='right'): 72, Interval(0.74, 1.01, closed='right'): 1455} | FP total 1527

== US: n=182,670 true pairs=632,310  F0.5=0.98240
   retrieval misses 11,004 (1.74%)   pruning misses 463 (0.07%)
   remove all FP -> 0.98543 (+0.0030)   add all model-missed (in cands) -> 0.99107 (+0.0087)
   model FN by p: {Interval(0.0, 0.02, closed='right'): 2249, Interval(0.02, 0.1, closed='right'): 2835, Interval(0.1, 0.3, closed='right'): 4029, Interval(0.3, 0.54, closed='right'): 4382, Interval(0.54, 0.74, closed='right'): 3492, Interval(0.74, 1.01, closed='right'): 9}
   model FN with ce_p (in band): 11784 of 16996 | FN ce_p>0.5: 6802
   FP by p: {Interval(0.0, 0.02, closed='right'): 0, Interval(0.02, 0.1, closed='right'): 0, Interval(0.1, 0.3, closed='right'): 0, Interval(0.3, 0.54, closed='right'): 0, Interval(0.54, 0.74, closed='right'): 67, Interval(0.74, 1.01, closed='right'): 1902} | FP total 1969

--- India RETRIEVAL misses (S1 || true match) ---
  Raj Food Pvt Ltd ; Plot No.A-9, Street No.13, Ida Nacharam Hyderabad, Hyderabad, Telangana
    || Raj Foeod Pvt Ltd ; Plot No.a-c-9, Hyderabad, TG
  Eastern Infra Pvt Ltd ; 1, British Indian Street New Building, 6Th Floor, Suit No. 615, Kolkata, Howrah, West 
    || ইস্টার্ন ইনফ্রা প্রাইভেট লিমিটেড ; 1, KOLKATA, HOWRAH, পশ্চিমবঙ্গ
  City Foods Private Limited ; 5, 6, Floor-Grd, Plot-29B, 4, Mira Mansion, Sion Circle, Mumbai, Maharashtra
    || सिटी फूड्स प्राइवेट लिमिटेड ; 5, Mumbai, MH
  Al Consultants Private Limited ; 119 Bank Enclave, Lakshmi Nagar, East Delhi, Delhi
    || *** Al Consultants Private  Ltd ; Bank Enclave, East Delhi, DL
  North Consulting Limited ; Raigad, Maharashtra, Raigad, B-206, Gurukrupa App, Pl-17, Sect- 09, Kamothe
    || North कंसल्टिंग लिमिटेड ; 
  One Investment Private Limited ; No. 9 And 10, Pampa Extension, 1St Main, Bangalore North, Bangalore, Karnatak
    || ಒನ್ ಇನ್ವೆಸ್ಟ್‌ಮೆಂಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ ; Door No 9 And 10, Bangalore Cantonment, Bangalore, KA
  Tech Enterprises Private Limited ; Tk-1, Third Floor, Living Style Mall, Jasola, New Delhi, South Delhi, Delhi
    || tech enterprises private limited ; Tk-1, New Delhi, दिल्ली
  Balaji Finance Limited ; 27, Kilokari Village Opp. Thapar Business Centre, Sunlight Co, Lony, New Delhi, East 
    || Mr Balaji Finance ; 
  My Construction Private Limited ; Vinit Tower Chs Ltd, Flat No. 607, Cst No 1320 Ajv Link Rd Andheri West, Nr 
    || माय कंस्ट्रक्शन प्राइवेट लिमिटेड ; VINTI TOWER CHS LTD, MUMBAI, MUMBAI CITY, Maharashtra
  MC International Private Limited ; F No 3, No 15, 16Th Main Rd, 1St Stage, Btm Layout, Bangalore, Karnataka
    || MC Internati0na1 Private Limited (ID: 64721) ; Karnataka, F NO 3-, NO 15, 16TH MAIN RD, 1ST STAGE, BTM LAYOUT,
  Swastik Estate Private Limited ; Gat No 143, Lakhmapur Koshimbe Road, At Post Dahiwi, Tal Dindori, Nashik, Mah
    || Swastik Estate Private Ltd ; 
  Roopesh India ; 115, Abhay Khand-I Indirapuram, Ghaziabad, Uttar Pradesh
    || R0OPESH-INDIA ; 115, GHAZIABAD, उत्तर प्रदेश

--- India MODEL misses (in candidates, not predicted) ---
  p=0.326 ce=nan
  HSM Avenues Private Limited ; Fl No. 14 Mayureshwar, Apartment, Bhigwan Road, Baramati, Pune, Maharashtra
    || Brixcira ; FL NO. 14 MAYURESHWAR, Maharashtra, BARAMATI, PUNE
  p=0.630 ce=nan
  Aarvtech Bearings Private Limited ; Plot No. 10-11, Block No. 212, Shiv City Industries-4, Mankana, Tal. Kamre
    || AARVTECH BENGNIRSG-PRIVATE LIMITED ; 
  p=0.035 ce=nan
  Scorpius Realty Pvt Ltd ; Gat No. 345, Shop No. 01, At Avhe, Post Jambhud, Pandharpur, Solapur, Maharashtra
    || Scorpius Pvt Ltd Service ; 
  p=0.004 ce=nan
  Rds International Pvt Ltd ; Old No.6, New No.45 Muthurangam Salai, T. Nagar, Chennai, Tamil Nadu
    || Rds  Pvt Ltd Center ; 
  p=0.681 ce=nan
  Genesis Therapies Pvt Ltd ; Kamal Kunj, Ahinsa Circle, Subhash Marg, C-Scheme, Jaipur, Rajasthan
    || GENESIS TERDCAPISM PVT LTD ; 
  p=0.005 ce=nan
  Psc (India) Ayurveda Pvt Ltd ; Charai House No 26 Vill, Dakshin Khia Haipur, Dhangaon, East Midnapore, West Be
    || Psc (India) Pvt Pvt Ltd Partners ; 
  p=0.330 ce=0.14669740200042725
  Premier Services ; H.No. 8-2-293/82/B/125, Road No. 10C, Jubilee Hil, Khairatabad, Hyderabad, Telangana
    || Premier Services Company ; Road No. 10C, Jubilee Hil, 8-2-293/82/B/12, Khairatabad, TG, Hyderabad
  p=0.625 ce=0.7926247715950012
  Mars Marketing Private Limited ; Kamala Nilyam, Avinashi Road Peelamedu, Coimbatore., Coimbatore, Tamil Nadu
    || BRIXRIZARIZA ; KAMALA NILYAM, COIMBATORE, Tamil Nadu

--- India FALSE merges ---
  p=0.848 ce=0.85500568151474
  White Properties Private Limited ; A-124 S/F New Friends Col, Near Arya Samaj Road, New Delhi, South Delhi, De
    || Solbelolum Sys ; A-124 S/F New Friends, Colony Arya Samaj Road, New Delhi, South Delhi, Delhi
  p=0.857 ce=0.8767399191856384
  Hyderabad Services Private Limited ; 275, Narsing Heights, Narsingi, Rajendranagar, Rural, Hyderabad, Telangan
    || Hyderabad Pltfomr Pltfomr ; HN 6 275, NARSING HEIGHTS, NARSINGI, RAJENDRANAGAR, RURAL, HYDERABAD, Telangana
  p=0.821 ce=0.7745833992958069
  Jai Interior Private Limited ; D-504, 5Th Floor, Priyamvadha C.H.S, Vaithara Nagar Nahur Road, Mulund (West), 
    || Jai Interior-Private Limited ; 
  p=0.946 ce=0.9994093179702759
  Innovative Consulting Private Limited ; A-10, Sheetal Apartment, Vapil Silvasa Road, Vapi, Valsad, Gujarat
    || ઇનોવેટિવ કન્સલ્ટન્સી પ્રાઇવેટ લિમિટેડ ; Gujarat, VALSAD, SILVASSA ROAD, VAPI
  p=0.928 ce=0.40697982907295227
  Nv Services Limited ; Office No. B-1, 2Nd Floor, B-156, New Ashok Nagar, East Delhi, Delhi
    || NV Ltd Services ; 
  p=0.979 ce=0.9992648959159851
  Hyderabad Construction Limited ; 12-1-331/C/1, Dattatreya Colony Asif Nagar, Hyderabad, Telangana
    || Hyderabad Limited Center ; 12-1-331/C/1, Dattatreya Colony Asif Nagar, Hyderabad, TG
  p=0.979 ce=0.7365901470184326
  Foundation Tech Private Limited ; A-203 Ashok Nagar, Delhi, North Delhi, Delhi
    || Foundation Tech Limited ; दिल्ली, A-203 Ashok Nagar, Delhi, North Delhi
  p=0.792 ce=nan
  Svr Law Chambers ; Khasra No 38, Vaishali Gali No. 1 Dabri, Palam Road, Delhi, East Delhi, Delhi
    || VMN LAW CHAMBERS ; 
```
