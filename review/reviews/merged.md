# Merged review — Opus × Fable 교차 검토 종합

- 대상: `review/report.md` (수정하지 않음)
- 입력: `reviews/review-opus.md`, `reviews/review-fable.md`
- 병합·판정: Claude (Fable 5.1, 이 세션). 한쪽만 지적한 항목과 충돌 항목은 **내가 직접 1차 출처를 열어** 확인했다. 열어본 출처는 §5, 끝내 확인하지 못한 항목은 §6에 있다.
- 표기: ✅ 맞음(내가 1차 출처로 확인) / ❌ 틀림(출처가 리뷰어 주장과 반대) / ⚠️ 부분 / ❔ 미확인(검색 스니펫 수준)

---

## 1. 두 리뷰가 모두 지적한 항목 (신뢰도 높음)

### 1-A. 서지·수치 오류 (report의 문장 → 정정)

| # | report.md 위치 / 문장 | 정정 | 근거 |
|---|---|---|---|
| A1 | TL;DR 3번째 불릿, Thread 5 FwdLLM 행, H3 근거: "INT4/GPTQ 양자화 LLaMA-7B peak **1.5GB**" | 논문(USENIX ATC'24) Table 7의 LLaMA-7B INT4(GPTQ) peak memory는 **4.0 GB**(FP16 15.6, INT8 7.9). 측정은 **AGNEWS 분류 과제**, Pixel 7 Pro, **연합학습** 설정이며 NPU 속도는 "we emulate its speed"로 **추정치**. 1.5GB는 GitHub README 한 줄에만 있고 양자화 조건이 없으며 LLaMA 코드는 "future work". "10분"은 INT4 조건이 맞음(CPU 0.19h≈11분). | 내가 PDF 텍스트로 Table 7 재확인 ✅ |
| A2 | Thread 5 MobileFineTuner 행, Caveats: "MobiSys 2026 채택으로 보고됨" | MobiSys 2026 accepted list에 **없음**(온디바이스 fine-tuning 논문은 FBLayout). 공식 README 2026-09-01: "conditional accepted by **Sensys2027**". → "arXiv preprint(README 기준 SenSys 2027 conditional accept)"로 표기. v1/v2 제목이 다르니 인용 버전 명시. | 두 페이지 모두 내가 재확인 ✅ |
| A3 | Thread 1 마지막 행: PRPF + "Do Proactive Agents Really Need an LLM…"을 한 행에 묶고 "기존 benchmark(ProactiveAgent, FingerTip) 재사용" | PRPF(Xiaomi, arXiv 2606.03236)는 **ProactiveMobile**에서만 평가. ProactiveAgent + FingerTip-20K는 TGL 논문(Liu et al., arXiv 2605.30152)만 사용. **행을 둘로 분리**하고 저자·벤치마크를 각각 기재. TGL 수치: 14개 backbone 전부 F1 향상(mean +16.7, up to +46.0), LLM-as-trigger 대비 4–7×/12–83× 빠름, ~220 MiB BF16 on-device. | 2605.30152 abstract 내가 재확인 ✅ |
| A4 | Caveats: "CIPHER의 교정 비용 감소율(요약 31%, 이메일 73%)은 2차 요약 출처에서만 확인" | 1차 출처(arXiv 2404.15269 v3 본문)에 그대로 있음 → **Caveat 삭제**. 비교 기준은 **no-learning 대비**(Table 2: 요약 32,974 vs 48,269 → −31.7%; 이메일 8,391 vs 31,103 → −73.0%). 단 **oracle 대비 격차가 큼**(요약 6,573, 이메일 1,851 → 4.5–5배) — H1 리스크로 추가. NeurIPS 2024 poster 확인. | Table 2 내가 재확인 ✅ (충돌 C1 참조) |
| A5 | Thread 2 ASTRA-bench 행, Caveats: "주인공 수 표기 불일치(4 vs 5)" | **원문 자체의 모순**: abstract "four protagonists", 본문은 5명을 이름으로 나열(Dawei Shen, Theo Appleseed, John Quinn, Emily Rose, Lucas Garcia). 코드도 abstract "released", 본문 `github.com/<coming-soon>`. → "본문 기준 5명, abstract와 상충"으로 확정하고 Caveat에서 제거. 시나리오 수는 abstract·§5.1·Table 2 모두 **2,413**. 나머지(평균 14일, CC BY-NC-ND 4.0, Claude-4.5-Opus 0.9112, payload 병목)는 맞음. | HTML 내가 재확인 ✅ (충돌 C2 참조) |
| A6 | Thread 5 CI-Bench 행, Caveats "CI-Bench 저자 목록" | 저자: Zhao Cheng, Diane Wan, Matthew Abueg, Sahra Ghalebikesabi, Ren Yi, Eugene Bagdasarian, Borja Balle, Stefan Mellem, Shawn O'Banion. **44,100**은 본문 Dataset 절에 명시(abstract는 "44 thousand"), 8 domains, "self-contained and stateless" 명시. → Caveat 삭제. AirGapAgent와 같은 그룹의 연속 연구. | HTML 내가 재확인 ✅ |
| A7 | Thread 1 FingerTip 행 "(arXiv 2507.21071, 2025)" | **ICLR 2026 poster**, "20K unique human demonstrations". | 두 리뷰 일치 |
| A8 | Thread 2 PersonaBench 행 (venue 없음) | **Findings of ACL 2025**. (+ 규모 오류는 §2-F1) | 두 리뷰 일치 |
| A9 | Thread 5 "SAPA-Bench / PrivacyBench 계열 (2025)" | SAPA-Bench = "Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents"(arXiv 2508.19493), GitHub에 **AAAI 2026**, 7,138 시나리오. PrivacyBench(arXiv 2512.24848)는 **대화형 RAG 비서**의 비밀 유출 벤치마크(Mukhopadhyay et al.)로 **별개 연구** → "계열"로 묶지 말 것. "민감도 3단계"는 abstract·README에 단계 수가 없음(❔). | 내가 abstract·README 재확인 ✅ |
| A10 | Thread 5 "LoRA Peak Memory Reduction … 26.20GB → 1.02GB" → "3B 야간 LoRA가 수치상 가능"; H3 동일 | 제목 "Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices"(Dbouk, Reisser, Mandke, Navali, Louizos; Qualcomm). Table 1의 26.20→1.02GB는 **NVIDIA A100에서 프로파일링**, FP32+LoRA baseline 대비 INT4 base + memory-efficient checkpointing(디스크 offload) + softmax 근사 + logits masking, batch 1, rank 16. 실기기 수치는 **§4.6 Table 8(2025년 12GB 폰)** 에 따로 있음 → 그 표를 인용해야 함. | HTML 내가 재확인 ✅ |
| A11 | Thread 4 Attelia 행 "통제 실험, 30명×16일 in-the-wild … 인지 부하 46%↓. 민감 사용자에서 좌절 28%↓" | 세 출처가 한 셀에 섞임. PerCom'15 abstract: **통제 실험 46%↓**, **in-the-wild 30명×16일 33%↓**(인지 부하). 좌절 28%(민감 사용자, in-the-wild)는 PMC 2016 저널판("Towards attention-aware adaptive notification on smart phones") 기준(❔ 스니펫). Attelia II(UbiComp'15)는 다기기 확장으로 "UI-event-only 대비 71.8% 더 큰 workload 감소"(❔). → 수치별 출처 분리. | PerCom abstract 내가 재확인 ✅ (충돌 C5) |
| A12 | Thread 5 P³ "(SIGIR 2026)" | arXiv 페이지에 venue 없음. 검색 결과는 SIGIR '26(Melbourne) 채택으로 표시(❔). 수치 90.3–95.7%, leakage +1.5–3.5%는 abstract 확인. | 두 리뷰 일치 |
| A13 | Thread 4 AHs 2026 행 | 제목·DOI 10.1145/3795011.3795067 존재. 저자 Lingler, Frijns, Boess, Murziakova, Wintersberger. abstract: **N=21**, 사무실 환경, "providing the model with the history of users' previous ratings led to the strongest improvements", "removing all contextual features resulted in the best overall alignment". "각 1–7" 척도 3항목은 abstract에 없음(❔). Key Finding 5는 "N=21 단일 연구"로 완화. | S2 API abstract 내가 확인 ✅ |
| A14 | Key Finding 2: Mialon 인터뷰 인용 | 논문 본문에 1차 수치 있음: GPT-5(high) pass@1 **42.1**, Time **0.0 → instant 모드 34.4**, "trade Time performance for Execution performance due to longer thinking". 인터뷰 대신 논문 인용. Kimi-K2는 **20.1**(abstract 21%는 반올림; "출처마다 다름" 문구 삭제). 1,120 = 800 + 320, 10 universes, 12 apps/101 tools, verifier 0.99 P / 0.95 R on 450 trajectories, "temporal consistency across apps" 한계 모두 확인. | HTML 내가 재확인 ✅ |

### 1-B. 누락 선행연구 (두 리뷰 모두 지적, 내가 abstract 재확인 ✅)

| # | 논문 | 어느 표 | 왜 중요한가 |
|---|---|---|---|
| B1 | **ProPerSim** — Kim et al., **ICLR 2026**, arXiv 2509.21730 | T1+T3 | 32 personas 시뮬레이션에서 proactive 제안 + 피드백으로 "steadily improves user satisfaction" → **proactive + 선호 학습 곡선(Δt=✓)** 을 이미 결합. gap (d) 부분 잠식. |
| B2 | **EOPA** — Wang et al., arXiv 2608.04416 | T1/T3/T5, H1·H3 | 온라인 피드백으로 개입 타이밍 선호를 **재학습 없이** 갱신, timing F1 **+19.80**, 일일 적응 11.41s→0.39s. H3(야간 재학습)의 반례이자 H1의 직접 근거. |
| B3 | **Proactive Service Agents: A Unified Decision Framework…** — Tang et al., arXiv 2609.03727 | T1, (b)·(c) | **silent / ask / assist / act** 4지 행동, "option value of waiting", interruption cost, POMDP → 보고서 4-way 행동 공간과 거의 같은 프레임. 차이(마감 함수, digest, 실행 검증, 교정)를 명시해야 함. |
| B4 | **π-Bench** — Zhang et al., arXiv 2605.14678 | T1 | 100 multi-turn tasks × 5 personas, 숨은 의도·세션 간 연속성. |
| B5 | **Ask Now, Use Later (ATRBench)** — Wu et al., arXiv 2605.28108 | T3 | 8개 frontier가 oracle 대비 **≥62점** 낮음, 병목은 선호 **획득**. 저녁 digest 질문 설계·H1 반례. |
| B6 | **MyPCBench** — Jang et al., arXiv 2606.16748 | T2 | 17 앱, 184 tasks, Claude Opus 4.6 55.4%. iOSWorld 그룹의 데스크톱판. |
| B7 | **From Overwhelmed to Overview** — Chen et al., PACM HCI (MobileHCI) 2025, DOI 10.1145/3743703 | T4 | ChatGPT 알림 요약 앱 in-the-wild 1주 + **20명** 인터뷰. 우선순위 유형 3, 공개 수준 3. 본문이 batching 문헌([22])도 정리. `batch_to_digest`의 직접 HCI 선행. |
| B8 | **PersoNo** — Zheng et al., **ISMAR 2025**, arXiv 2508.19622 | T4, C1 | 개인화 알림 urgency 분류기(18명, 81.5%, FN 0.381). "activity context is equally important as the content and the sender". |
| B9 | **HorizonBench** — Li et al., arXiv 2604.17283 (report엔 ID만) | T3, H1 | 360 simulated users × 6개월, 25 모델 중 최고 **52.8%**, 모델이 갱신된 선호를 추적 못 함. |

### 1-C. Gap statement (두 리뷰 일치)
5요소 **완전 결합**으로는 살아남는다. 그러나 (i) proactive + 피드백 기반 선호 학습 곡선(ProPerSim, EOPA, "1,000 Personas"), (ii) 4지 행동·비용 프레임(Proactive Service Agents) 및 상용 urgency 순위화·일일 digest(Apple Priority Notifications, Samsung Now Brief, Pixel Magic Cue/Daily Hub), (iii) 다중 앱 합성 개인 데이터와 realism 검증(PersonaTrace, MyPCBench, Privasis)이 이미 존재한다. 따라서 **Key Finding 1** "batching·digest를 행동 옵션으로 두는 연구는 없다", **Key Finding 3** "realism을 실제 로그와 비교한 연구는 찾지 못했다", **(d)** "각각 따로 다뤘다"는 그대로 두면 리뷰어에게 반박당한다.

### 1-D. 가설 H1/H2/H3 (두 리뷰 일치)
- **H1**: P³는 선호 추출이 아니라 **원본 프로필을 로컬에 둔 RAG 수정** 방식 → H1 근거에서 빼고 H2/프라이버시 근거로 이동. CIPHER는 문체 선호이며 oracle 대비 격차 큼. 반례: Ask Now Use Later, KnowU-Bench("preference acquisition" 병목), PrefEval, HorizonBench. → "선호 획득 정책"을 H1 평가에 넣을 것.
- **H2**: 인용 근거가 약하거나 오용됨(AirGapAgent는 프라이버시 분리, ProAgentBench 계층 분해는 과제 정의). 가장 강한 근거는 TGL(2605.30152)이나 **개인화되지 않은 단일 trigger·offline 평가**. → 판정을 "효율·오탐 억제는 강함, 개인화된 urgency 판단은 미검증"으로.
- **H3**: 핵심 수치(FwdLLM 1.5GB, MobileFineTuner MobiSys)가 틀림. 반례 EOPA. → 근거 교체(§4 P2-23) 후 "ablation의 한 arm"으로 축소.

### 1-E. 벤치마크 (c) (두 리뷰 일치)
- `urgency` 등급과 `deadline_t`/`cost_of_delay` 중복 → 등급은 파생 변수로.
- 단일 정답 행동 대신 (persona, context)별 **허용 집합 또는 비용 행렬**.
- Net utility λ 고정 → λ sweep / Pareto.
- ECE의 확률 원천 미정의.
- 인간 검증: 제3자 annotator α는 persona 당사자 urgency를 못 잼; **참가자가 합성 이벤트를 겪지 않으므로 ESM과 합성 라벨의 상관은 계산 불가** → WoZ replay 또는 실제 알림 라벨링; 연구실 내 5–10명은 IMWUT/CHI 기준 미달(선행: My Phone and Me 20, Pielot 24, AHs 21, Attelia 30, Fitz 237); 비공개 데이터는 NeurIPS D&B 재현성과 충돌.
- 시뮬레이터 의존성: 복수 시뮬레이터 LLM 보고("Lost in Simulation" 최대 9pp).
- Attelia breakpoint는 UI 이벤트/활동 전환 경계 → 합성 캘린더·메일 타임라인에는 없는 신호.

---

## 2. 한쪽만 지적한 항목 — 내가 직접 확인

### 2-O. Opus만 지적

| # | 지적 | 판정 | 내가 확인한 내용 |
|---|---|---|---|
| O1 | FwdLLM "10분"은 NPU 에뮬레이션·분류 과제 조건 | ✅ | Table 7 각주 "LLaMA currently is not supported by mobile NPU, therefore we emulate its speed". CPU INT4 0.19h. |
| O2 | CIPHER oracle 대비 격차(요약 32,974 vs 6,573; 이메일 8,391 vs 1,851) | ✅ | Table 2 확인. |
| O3 | MobileFineTuner 스케줄러는 **저배터리 throttling**이지 야간 충전 스케줄링이 아님 | ✅ | §4.2: 배터리가 임계값 μ 아래로 떨어지면 빈도를 ρ만큼 줄임. "at night"는 §5.3 사례 연구의 설계 선택. → 시사점 문구 수정. |
| O4 | BehaviorGen 62.0/87.8/MIA<0.55가 abstract에 없어 의심 | ❌ (report가 맞음) | 본문 Table 2(Bert4Rec backbone): Tencent 62.0%, Smartphone 87.8% replacement rate; A.5: MIA 성공률 Smartphone <0.55, Tencent <0.74; 생성기 gpt-4o-2024-0806. → 조건만 추가. |
| O5 | "Tell Me Why You're Asking"(CHI 2026), "Read the Room"(CHI EA 2026) could not verify | ❌ (report가 맞음) | 둘 다 존재: DOI 10.1145/3772318.3790950(33명 인터뷰), DOI 10.1145/3772363.3798392("…When LLM Agents Should Participate in Everyday Mixed-Context Group Chats…", 6 dyads). → DOI만 추가. |
| O6 | AHs 2026 N=21·"평정 이력" 미확인 | 해소 ✅ | §1 A13. |
| O7 | MCP-Persona 세부(24 서버, Notion, email, 173/970, 91.5%) 미확인 | ✅ report 맞음 | HTML: 24 MCP servers(12 personalized), Notion·Obsidian·Gmail·163-Email·Lark·Slack…, 173 tasks, 970 checkpoints, 91.5% alignment, 전화번호 등 민감 필드 가짜로 대체. |
| O8 | PARE 소속(UW)·Stackelberg POMDP 미확인 | ✅ report 맞음 | HTML 1쪽: UCSB / Apple / U. Washington. "We formalize this as a Stackelberg POMDP"(App. D). GitHub deepakn97/pare 존재. |
| O9 | SAPA-Bench·PrivacyBench 분리, AAAI 2026 | ✅ | §1 A9. |
| O10 | Gaia2 인터뷰 → 논문 수치로 교체 | ✅ | §1 A14. |
| O11 | Pielot CHI'14 수치(6.15분, 81.2%)는 1차 출처와 일치 | ✅ | 저자 PDF: 24명, 2주, 중앙값 6.15분, accuracy 70.6%, precision(fast attendance) 81.2%. 표에 accuracy·N·기간 추가 권장. Fable의 의심(§2-F 참조)도 해소. |
| O12 | My Phone and Me 20명/10,372/474 | ✅ | 저자 PDF: 74명 설치 → 설문 14회 이상 응답 **20명**, 10,372 알림, 474 설문, 11 성격검사. |
| O13 | 누락: PersonaTrace(EACL 2026 Industry) | ✅ | 프로필→이벤트 시퀀스→이메일·메시지·캘린더·리마인더 artifact, real OOD 과제로 realism 검증. KF3 반례. |
| O14 | 누락: ProMemAssist(UIST 2025) | ✅ | Pu et al.(Meta 연구진), "timing predictor that balances the value of assistance with the cost of interruption", 12명. Net utility 선행 정식화. |
| O15 | 누락: AcCoRD(arXiv 2608.27818) | ✅ | 선호가 "formed, revealed, adjusted, and relaxed"; 5 LLM 모두 mid-interaction 선호 변화에 취약. C3 drift 축 근거. |
| O16 | 누락: PPP/UserVille(COLM 2026, arXiv 2511.02208) | ✅ | Sun, …, Neubig, Sap, Yang. 설정 가능한 LLM 사용자 시뮬레이터, preference adherence 보상 RL, GPT-5 대비 +16.7. |
| O17 | 누락: Lost in Simulation(arXiv 2601.17087) | ✅ | 시뮬레이터 LLM에 따라 성공률 최대 **9pp** 변동, 난이도별 편향. 방법론 위협. |
| O18 | 누락: XPerT(MobiSys 2025, arXiv 2504.13938) | ✅ | Wang, Yang, Yin, Gao. 온디바이스 개인화 계산 −83%, 데이터 효율 +51%. H3 보강. |
| O19 | 누락: Apple AFM Tech Report 2025(arXiv 2507.13575) | ✅ | 3B on-device, 2-bit QAT, Foundation Models framework에 LoRA adapter fine-tuning. |
| O20 | 누락: MobileRLHF(README) | ✅ | README 2026-07-10: "reinforcement-learning-based preference post-training on smartphones in both standalone and federated settings"(논문 없음, README 수준). |
| O21 | 누락: Forget to Improve(arXiv 2606.25115) | ⚠️ | 가중치 갱신 없는 예산 기반 메모리 큐레이션은 맞으나 testbed가 **Jetson 로봇**이지 스마트폰이 아님. 인용 시 조건 명시. |
| O22 | 누락: Sommuang et al. LLM-agent EMA 합성(arXiv 2508.02679) | ✅ | StudentLife 기반 에이전트가 EMA 자기보고를 생성. Caveat "LLM 합성 ESM 없음"은 IMWUT/UbiComp로 범위 한정 필요. |
| O23 | CI-Bench 정밀 수치 44,100이 abstract에 없다 | ⚠️ | abstract는 "44 thousand"이나 본문에 44,100 명시 → Fable·report가 맞음. |

### 2-F. Fable만 지적

| # | 지적 | 판정 | 내가 확인한 내용 |
|---|---|---|---|
| F1 | PersonaBench "15 characters" → **1,515** | ✅ | HTML: "The test set includes 1515 characters"; 582 = 269 basic + 186 preference + 127 social; GPT-4o gold-context recall 0.444; 선호 갱신 <1%; noise 0/0.3/0.5/0.7. |
| F2 | KnowU-Bench 지표는 **Act / Silent / Stop rate**; "consent"는 지표 아님 | ✅ | HTML Metrics 절 확인. 42/86/64 tasks, 11 models, Sonnet 4.6 실패의 80.0%가 intervention/passivity. → Thread 1 라벨 열과 (b) 2순위 문구 수정. |
| F3 | iOSWorld 최고 51.9%/36.7%는 **Claude Opus 4.6**, Qwen3.5-35B-A3B는 10.5% | ✅ | HTML 확인(26 apps, 133 = 27/60/46, Jordan Avery, κ=0.77). |
| F4 | Gaia2 Kimi-K2 20.1 | ✅ | §1 A14. |
| F5 | MobileFineTuner "1B당 16GB / 폰 RAM 4–16GB"가 abstract에 없어 의심 | ❌ (report가 맞음) | §4 서두에 그대로 있음: "16 GB of RAM is required per 1 billion parameters for FP16 training … typical mobile phones in 2025 are equipped with only 4–16 GB of RAM". |
| F6 | ProAgentBench 3단계 프라이버시 파이프라인 미확인 | ✅ report 맞음 | HTML: VLM-based Preliminary Judgment(Qwen3-VL-Plus) → Volunteer Correction → Rule-based Filtering. 28,528 events(7,222 LLM-related), 500+h, B=0.787 vs synthetic 0.166, real 74.0% vs synthetic 62.1%(LLaMA). 코드는 anonymous.4open.science. |
| F7 | SentinelBench GitHub 미확인 | ✅ report 맞음 | github.com/microsoft/sentinel_environments 존재(MIT, 10 앱 × 10 tasks). |
| F8 | Attelia II in-the-wild 71.8%; 30×16일·28% 미확인 | ⚠️ | 30명×16일과 33%는 **PerCom'15** in-the-wild(abstract 확인 ✅). 28%는 PMC 2016 저널(❔). 71.8%는 UbiComp'15(❔). |
| F9 | Kruger CHIRA 2025 / PersonaMem COLM 2025 / PersonaLens Findings ACL 2025 | ✅ | arXiv 2509.13892 comments "CHIRA 2025"; PersonaMem GitHub "[COLM 2025]"; PersonaLens aclanthology 2025.findings-acl.927. → report의 "PersonaMem, PersonaLens (Findings ACL 2025)"는 PersonaMem venue를 잘못 묶음. |
| F10 | 누락: "After Talking with 1,000 Personas"(arXiv 2602.04000) | ✅ | Xuan et al. 1,000 persona 시뮬레이션으로 timing·autonomy·style 선호 학습, "lightweight activation-based steering" + "on-device and privacy constraints". (34명 인간 실험 수치는 abstract에서 미확인 ❔) |
| F11 | 누락: Fitz et al. 2019(Computers in Human Behavior) | ✅ | 무작위 현장실험(India, n≈237 ❔스니펫 / "over 200"), 3회/일 batching은 스트레스↓·집중↑·잠금해제↓, 매시간 batching은 효과 없음, 완전 차단은 불안↑. `batch_to_digest` 빈도 설계 근거. |
| F12 | 누락: Privasis(arXiv 2602.03183) | ✅ | 1.4M 합성 개인 기록, 55.1M 속성, 의료·법률·금융·캘린더·문자. |
| F13 | 누락: ToolSandbox(Apple, arXiv 2408.04682) | ✅ | stateful tools + built-in user simulator + milestone/minefield. |
| F14 | 누락: UserBench(arXiv 2507.22034) | ✅ | 최고 모델도 선호의 <30%만 발견, 완전 정렬 20%. (UserToolBench 2608.10042는 미개봉 ❔) |
| F15 | 누락: PSPA-Bench(arXiv 2603.29318) | ✅ | 12,855 개인화 지시, 10 시나리오, 22 앱, 11 agents. |
| F16 | 누락: FBLayout(MobiSys 2026) | ✅ | accepted list에 "FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs"(Tam et al.). 본문 미개봉 → 수치 인용 전 확인 필요. |
| F17 | 누락: Liu et al. "Sensing What Surveys Miss"(CHI 2026, arXiv 2602.00880) | ⚠️ | 존재·CHI 2026 확인. 다만 **설문 응답 지원**(EDA+마우스, 32명) 맥락이라 폰 알림과는 거리 있음 → 선택 인용. |
| F18 | 상용 기능: Apple Priority Notifications(iOS 18.4), Samsung Now Brief(One UI 7), Pixel 10 Magic Cue/Daily Hub | ✅(언론·위키·공식 블로그) | iOS 18.4 Priority Notifications는 Apple Intelligence로 "important and time-sensitive" 알림을 별도 섹션에; Now Brief는 One UI 7 도입 일일 요약 카드; Magic Cue는 Gemini Nano(Tensor G5) 온디바이스(hybrid) 선제 제안 + Daily Hub. 논문 아님 → "제품 수준 선행"으로 한 문단. |
| F19 | 행동 집합에 "ask(지금 물어보기)" 없음 | ✅(설계 판단) | Proactive Service Agents(silent/ask/assist/act)와 PARE(Observe 모드 제안)가 ask를 별도 옵션으로 둠. 추가하거나 배제 이유 명시. |
| F20 | Pielot 6.15분/81.2% 미확인, "accuracy 70.6%"일 수도 | 해소 ✅ | §2-O11: 둘 다 맞음(accuracy 70.6%, precision 81.2%, 중앙값 6.15분). |

---

## 3. 서로 충돌하는 지적 — 판정

| # | 쟁점 | Opus | Fable | 판정 |
|---|---|---|---|---|
| C1 | CIPHER 31%/73%의 비교 기준 | no-learning 대비 | "최강 baseline ICL-edit 대비" | **Opus가 맞음.** Table 2로 계산하면 no-learning 대비(요약 1−32,974/48,269=31.7%, 이메일 1−8,391/31,103=73.0%). ICL-edit 이메일은 32,405로 no-learning보다 나쁨. |
| C2 | ASTRA-bench 시나리오 수 | 2,413 | "본문 2,400 vs abstract 2,413" | **2,413으로 표기.** abstract·§5.1·Table 2 모두 2,413. "2,400"은 본문 서술 문장의 반올림으로 보이며 불일치로 쓸 사안 아님. |
| C3 | MobileFineTuner venue | "SenSys 2027 conditional accept(README)" | "MobiSys 2026 목록에 없음 → arXiv preprint" | **둘 다 맞고 양립.** "arXiv 2512.08211 (README 기준 SenSys 2027 conditional accept, 미확정)"로 표기. |
| C4 | H2 판정 | "근거가 가장 강함"은 과장 | 동의하되 근거를 TGL로 교체 | **Opus 문구 채택 + Fable 근거 채택.** 효율·오탐 억제 축은 TGL/PRPF로 강함, 개인화 urgency 판단 축은 미검증(TGL은 단일 체크포인트·offline). AirGapAgent는 유추 근거. |
| C5 | Attelia 수치 배치 | 46% 통제 / 33% in-the-wild(30×16일)는 PerCom'15 | 46%는 실험실; in-the-wild는 Attelia II 71.8%; 30×16·28% 미확인 | **Opus가 맞음(PerCom'15 abstract 확인).** 28%는 PMC 2016 저널의 "민감 사용자" 결과(스니펫), 71.8%는 UbiComp'15 별건. 셀을 3줄로 분리. |
| C6 | "Tell Me Why You're Asking" / "Read the Room" | 찾지 못함(삭제 권고) | (언급 없음) | **report가 맞음.** 둘 다 CHI 2026/CHI EA 2026에 존재(§2-O5). DOI만 추가. |
| C7 | SAPA-Bench "민감도 3단계" | 미확인 | 확인 | **미확인(❔) 유지.** abstract·README 모두 "sensitivity level" 주석만 언급하고 단계 수 없음. 본문 확인 필요. |
| C8 | LoRA 1.02GB 실기기 여부 | 조건·기기 미확인 | 12GB 폰 검증 | **둘 다 부분 맞음.** Table 1은 A100 프로파일링, §4.6 Table 8이 12GB 폰 실측. report는 Table 8 수치를 써야 함. |
| C9 | ProactiveMobile CVPR 2026 | 미확인 | GitHub 제목으로 확인 | **Fable.** xiaomi-research/proactive-mobile 저장소 제목에 "(CVPR 2026)". arXiv 페이지에는 없음 → "CVPR 2026 (GitHub 기준)". |
| C10 | BehaviorGen / MobileFineTuner 경험칙 / MCP-Persona / PARE / ProAgentBench 파이프라인 등 "abstract에 없음" 계열 의심 | 각각 한쪽이 의심 | — | **모두 report가 맞음**(본문에 있음). 리뷰어가 abstract만 보고 의심한 사례. 조건만 보강. |

---

## 4. report.md 최종 수정 목록 (우선순위순, 섹션·문장 단위)

### P0 — 사실 오류 (리뷰어가 즉시 잡을 것)
1. **TL;DR 3번째 불릿 + (e) H3 근거 1 + Thread 5 FwdLLM 행**: "INT4/GPTQ 양자화 LLaMA-7B peak 1.5GB" → "논문 Table 7: LLaMA-7B INT4(GPTQ) peak 4.0GB, AGNEWS 분류, Pixel 7 Pro, 연합학습; NPU 속도는 에뮬레이션. README는 조건 없이 1.5GB를 주장하나 논문에 없음". TL;DR에서 FwdLLM과 LoRA 1.02GB를 한 괄호에 섞은 부분 분리.
2. **Thread 5 MobileFineTuner 행 + Caveats 2번째 불릿**: "MobiSys 2026 채택으로 보고됨" → "arXiv 2512.08211 v2; README 기준 SenSys 2027 conditional accept(미확정)". 시사점 "야간 충전 중 학습 스케줄링" → "저배터리 시 빈도 감소(throttling) 스케줄러; 야간 충전 스케줄링은 본 과제가 설계".
3. **Thread 1 마지막 행**: PRPF(Ding et al., Xiaomi; ProactiveMobile 평가)와 TGL(Liu et al., 2605.30152; ProactiveAgent+FingerTip-20K; F1 +16.7 mean, 4–83× 빠름, ~220 MiB)로 **행 분리**.
4. **Thread 2 PersonaBench 행**: "582 문항, 15 characters" → "582 문항(269/186/127), **1,515** test characters"; venue "Findings of ACL 2025".
5. **Thread 2 iOSWorld 행 모델 열**: "Qwen 3.5 35B 등" → "최고 Claude Opus 4.6 (51.9% / multi-app 36.7%); Qwen3.5-35B-A3B 10.5%".
6. **Thread 1 KnowU-Bench 행 라벨 열 + (b) 2순위 첫 불릿**: "intervene / consent / silence, 거절 후 자제" → "Act rate / Silent rate / Stop rate(거절 후 중단)"; "consent 라벨 차용" 문구 삭제.
7. **Thread 4 Attelia 행**: 규모·결과 열을 "PerCom'15: 통제 실험 인지 부하 46%↓; in-the-wild 30명×16일 33%↓ / PMC 2016: 민감 사용자 좌절 28%↓ / UbiComp'15 Attelia II: 다기기, UI-event-only 대비 71.8% 추가 감소"로 분리. 라벨 정의 열의 "인지 부하"는 NASA-TLX(통제 실험) vs 체감 workload(in-the-wild)로 구분.
8. **Thread 5 LoRA Peak Memory 행 + H3 근거 3**: 제목·저자 추가; "26.20GB→1.02GB"에 "FP32+LoRA baseline 대비, INT4+checkpointing/offload+softmax 근사+logits masking, batch 1, **A100 프로파일링**; 실기기 수치는 §4.6 Table 8(2025년 12GB 폰)" 조건 명시.
9. **Thread 5 SAPA-Bench/PrivacyBench 행**: SAPA-Bench("Mind the Third Eye!", AAAI 2026, 7,138)와 PrivacyBench(대화형 RAG 비밀 유출, 2512.24848)를 분리하거나 후자 삭제. "민감도 3단계"는 확인 전까지 제거.
10. **Thread 2 ASTRA-bench 행 + Caveats**: "주인공 수 표기 불일치(4 vs 5)" → "본문 5명(이름 명시), abstract는 four로 상충"; 코드 "abstract는 released, 본문 링크는 coming-soon". Caveat 목록에서 제거.
11. **Caveats 3번째 불릿(CIPHER) 삭제**, 2번째 불릿의 "CI-Bench 저자 목록" 삭제(§1 A6 저자 기재). Thread 3 PRELUDE 행 지표 열에 "no-learning 대비 edit −31%(요약)/−73%(이메일), oracle 대비 4.5–5배 잔여"를 추가.
12. **Thread 2 Gaia2 행**: "Kimi-K2는 … 출처마다 다름" → "Kimi-K2 20.1%". **Key Finding 2**: Arize 인터뷰 인용 → 논문 본문(Time 0.0 → instant 34.4; 42.1 pass@1) 인용으로 교체.
13. **venue 보완**: FingerTip 20K "ICLR 2026 poster"; ProactiveMobile "CVPR 2026(GitHub 기준)"; Kruger "CHIRA 2025"; PersonaMem "COLM 2025"(PersonaLens와 분리); HorizonBench 제목·저자; P³ "SIGIR 2026(accepted, 검색 기준)"; ProactiveBench ICLR 2025·ContextAgent NeurIPS 2025는 유지.
14. **Thread 4**: "Tell Me Why You're Asking"에 DOI 10.1145/3772318.3790950(33명 인터뷰), "Read the Room"에 DOI 10.1145/3772363.3798392(6 dyads) 추가. AHs 2026 행에 저자·"removing all contextual features resulted in the best overall alignment" 추가, "(각 1–7)"은 확인 전까지 제거. Pielot 행에 "24명·2주, accuracy 70.6%" 추가. My Phone and Me는 그대로.
15. **보조표 BehaviorGen 행**: "Bert4Rec backbone 기준(Table 2); MIA <0.55는 Smartphone 데이터셋(Tencent <0.74); 생성기 gpt-4o-2024-0806" 조건 추가.
16. **Thread 1 PARE·MCP-Persona·ProAgentBench 행**: 그대로 유지(검증 완료). PARE GitHub·Stackelberg POMDP 표기 유지.

### P1 — Gap statement·Key Findings·표 보강
17. **결론 첫 문단, TL;DR 1번째 불릿, Key Finding 1·3, (d)**: "각각 따로", "batching·digest를 행동 옵션으로 두는 연구는 없다", "realism 비교 연구 없음"을 다음으로 교체 — "proactive 타이밍 + 피드백 학습 곡선(ProPerSim ICLR'26, EOPA, 1,000 Personas), 4지 행동·대기 옵션가치 프레임(Proactive Service Agents), 상용 urgency 순위화·일일 digest(Apple Priority Notifications, Samsung Now Brief, Pixel Magic Cue), 다중 앱 합성 개인 데이터와 realism 검증(PersonaTrace, MyPCBench, Privasis)은 존재한다. 그러나 마감·지연비용으로 정의한 urgency와 digest batching을 **행동으로 채점**하고, 대리 실행을 **환경 상태로 검증**하며, 교정 학습 곡선을 **같은 타임라인 위에서 온디바이스 제약 하에** 재는 폐루프 benchmark는 없다."
18. **표 행 추가** — T1: ProPerSim, EOPA, "1,000 Personas", π-Bench, Proactive Service Agents(프레임워크), ProMemAssist(UIST'25). T2: MyPCBench, PSPA-Bench, ToolSandbox. T3: Ask Now Use Later, AcCoRD, PPP/UserVille(COLM'26), UserBench, HorizonBench(정식 행). T4: Fitz 2019, Chen et al. MobileHCI'25, PersoNo(ISMAR'25); Liu et al. CHI'26은 선택. T5: XPerT(MobiSys'25), FBLayout(MobiSys'26), Apple AFM 2025, MobileRLHF(README), Forget to Improve(Jetson 조건 명시). 보조표: PersonaTrace(EACL'26), Privasis, Sommuang(EMA 합성). 방법론: Lost in Simulation.
19. **Caveats 4번째 불릿**: "IMWUT·UbiComp에서 LLM 합성 ESM … 찾지 못했다"에 "arXiv에는 Sommuang et al. 2025(EMA 합성)가 있음"을 덧붙여 범위 명시.
20. **(b) 2순위 KnowU-Bench**: 라벨명 수정(6번) 외에 "profile 숨김" 설계는 UserBench·UserToolBench와 같은 계열임을 한 줄 언급.

### P2 — 가설 (e)
21. **H1**: P³ 불릿을 H2/프라이버시로 이동. CIPHER 불릿에 oracle 격차 추가. 근거에 EOPA(재학습 없는 온라인 갱신)·Forget to Improve(예산 내 메모리) 추가. 리스크에 Ask Now Use Later(−62점), HorizonBench(52.8%), KnowU("preference acquisition" 병목) 추가. 권고에 "선호 **획득** 정책(언제 무엇을 물을지)"을 명시.
22. **H2**: 판정 문구를 "효율·오탐 억제 측면은 강함(TGL, PRPF); 개인화된 urgency 판단 측면은 미검증(TGL은 단일 체크포인트·offline)"으로. ProAgentBench 불릿은 "과제 정의상 분해"로, AirGapAgent 불릿은 "프라이버시 분리의 유추 근거"로 완화. 함의의 "수 MB~수백 MB"에 TGL ~220 MiB를 근거로 연결. AHs 2026 결과(맥락 특징 < 평정 이력)를 근거로 filter 입력에 개인 이력 필수임을 추가.
23. **H3**: 근거 4개를 "FwdLLM Table 7(4.0GB, 분류, 연합, NPU 에뮬레이션) / LoRA peak-memory §4.6 Table 8(12GB 폰) / PocketLLM(OPT-1.3B 6.5GB, OPPO Reno 6, 미분 없는 최적화) / XPerT(MobiSys'25) / FBLayout(MobiSys'26) / MobileRLHF(README) / Apple AFM(3B + LoRA adapter)"로 교체. 한계에 EOPA(재학습 없이 0.39s/일) 추가. 권고 설계의 "주 단위 LoRA 증류"는 근거 없음을 명시하고 ablation arm으로 격하.

### P3 — 벤치마크 (c)
24. **C1 라벨**: `urgency` 등급을 `deadline_t`·`cost_of_delay`에서 파생. 정답 행동을 (persona, user_context)별 허용 집합/비용 행렬로. 행동 집합에 `ask_now` 추가 여부 결정(Proactive Service Agents·PARE 인용). `interrupt_now(modality)`는 라벨이 아니라 제약 조건으로.
25. **C1 지표**: Deadline-met recall과 방해 여부를 독립 축으로; Time-to-action을 **지연 포함/instant 이중 모드**로(Gaia2 0.0 vs 34.4 인용); λ sweep/Pareto; ECE의 확률 원천(sampling k회 또는 verbalized confidence) 명시; 하루 방해 횟수는 이벤트 수로 정규화; modality별 방해 비용 정의.
26. **C2 지표**: `batch_to_digest` 항목의 `deadline_t` < digest 시각이면 미충족 채점 규칙; digest 시각(고정 저녁 vs breakpoint)을 변수/상수로 명시하고 Fitz 2019(3회/일 유효, 매시간 무효)를 빈도 근거로; Report fidelity·Privacy leakage의 judge와 인간 일치율 목표(ProactiveBench 91.80%, MCP-Persona 91.5%) 명시; CI 위반과 오수신자 전송(irreversible)을 분리.
27. **C3 지표**: Corrections-to-convergence의 censoring 규칙(cap·생존분석); 선호 **drift** 이벤트 추가(AcCoRD, HorizonBench); Repeat-mistake rate와 PVR 중복 정리; Forgetting/Over-generalization은 고정 probe set 일일 재평가로 정의; hard/soft 규칙별 PVR; 시뮬레이터 민감도 ablation 필수(PARE 방식, Lost in Simulation 인용).
28. **인간 검증 문단(C1) + Recommendations 3번**: (i) annotator에게 persona 프로필·선호 규칙을 제공하는 판정 절차와 α 목표치, (ii) 합성 사건을 참가자 폰에 주입하는 WoZ replay 또는 실제 알림에 같은 라벨 스키마 적용, (iii) 연구실 외부 ≥20–30명, IRB, 익명 집계 통계 공개, (iv) 복수 시뮬레이터 LLM 순위 안정성 보고.
29. **Thread 4 라벨 용어**: Pielot은 attentiveness(볼 수 있는가), Mehrotra CHI'16은 response time vs perceived disruption → C1에 attentiveness/receptivity 구분 추가. NASA-TLX는 Attelia 통제 실험 측정임을 Key Finding 5에서 명시. Attelia breakpoint 정의(UI 이벤트·활동 전환 경계) 정정 및 합성 UI 사용 로그 채널 필요성 명시.
30. **Recommendations 4번 baseline**: "규칙 기반(Attelia breakpoint + 우선순위 규칙)" 옆에 "경량 학습형 trigger(TGL 방식)"와 "EOPA식 온라인 파라미터 갱신"을 추가.

---

## 5. 병합자가 직접 연 출처 (검증에 사용)

arXiv abs/HTML: 2508.19493, 2512.24848, 2505.17615(html), 2606.19528(html×2), 2606.02470(html), 2604.00842(html), 2409.13903(html), 2502.20616(html), 2602.04482(html), 2604.08455(html), 2606.09764(html), 2602.04000, 2509.13892, 2602.03183, 2604.17283, 2602.00880, 2603.11955, 2504.13938, 2601.17087, 2404.15269v3(html), 2602.11964(html), 2603.01357(html), 2512.08211v2(html), 2507.21378, 2608.27818, 2511.02208, 2507.13575, 2606.25115, 2508.02679, 2507.22034, 2603.29318, 2608.04416, 2509.21730, 2609.03727, 2605.30152, 2508.19622, 2605.28108, 2606.16748, 2605.14678, 2408.04682
GitHub/학회/기타: Zhixin-L/SAPA-Bench, deepakn97/pare, microsoft/sentinel_environments, bowen-upenn/PersonaMem, Edge-Intelligence-Lab/MobileFineTuner README(raw), sigmobile.org/mobisys/2026/accepted_papers, neurips.cc/virtual/2024/poster/96078, aclanthology.org/2024.privatenlp-1.10, Semantic Scholar API(PerCom'15 Attelia abstract, AHs 2026 abstract), en.wikipedia.org/wiki/Galaxy_AI, communities.springernature.com(Fitz 2019 요약), okoshi.org/publication
PDF 로컬 추출(PyMuPDF): USENIX ATC'24 FwdLLM(Table 7), Mehrotra CHI'16(My Phone and Me), Pielot CHI'14, Chen et al. MobileHCI'25(NYCU 사본)
검색 스니펫만(❔): Attelia PMC 2016 28%/37%, Attelia II 71.8%, Fitz n=237, Apple iOS 18.4 Priority Notifications, Pixel Magic Cue, ProactiveMobile CVPR 2026(GitHub 제목), P³ SIGIR 2026

## 6. 끝내 확인하지 못한 항목 (저자 확인 필요)
- Attelia PMC 2016 저널의 28%(in-the-wild 민감 사용자)·37%(실험실 37명) — ScienceDirect/ResearchGate 403.
- Attelia II(UbiComp'15) 71.8% — ACM DL 403, S2 abstract elided.
- SAPA-Bench 민감도 단계 수(3?) — 본문 미개봉.
- P³ SIGIR 2026 채택 — accepted list 미개봉.
- "1,000 Personas" 인간 실험 34명 — abstract에 없음.
- AHs 2026의 3개 평정 항목·1–7 척도 — abstract에 없음.
- PRELUDE 저자의 "사용자별 fine-tuning은 costly … may even degrade" 인용(Opus) — 본문 위치 미확인.
- UserToolBench(2608.10042), Personal LLM Agents 서베이(2401.05459), FBLayout 본문 수치 — 미개봉.
- ContextAgent 온디바이스 크기 세부(report Caveat) — 두 리뷰 모두 다루지 않음.
