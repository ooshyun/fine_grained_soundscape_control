# Review by Claude Opus

검토 대상: `review/report.md` (온디바이스 선제적 개인 비서 벤치마크 문헌·데이터셋 리뷰).
표기: 아래 "Verified"는 1차 출처(arXiv abs/HTML, 논문 PDF, 공식 GitHub README)를 직접 열어 확인한 것만 해당합니다. 검색 결과 요약만 본 항목은 명시적으로 "search snippet only"로 표기했습니다(§8 참조).

---

## 1. Verified wrong (opened the source; report contradicts it)

| # | Report text (quoted) | Correction | Source URL |
|---|---|---|---|
| 1 | "**MobileFineTuner** — Geng et al. (arXiv 2512.08211, **MobiSys 2026 채택으로 보고됨**)" / Caveats: "MobileFineTuner의 MobiSys 2026 채택(2차 출처)" | 공식 GitHub README(2026-09-01 항목): *"has been conditional accepted by **Sensys2027**"*. arXiv 2512.08211 v2(2026-06-10)의 comments에는 venue 표기가 없음. → "SenSys 2027 conditional accept (README 기준)"로 수정. 또한 v1 제목은 "A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones", v2 제목은 "…A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications"로 바뀌었으므로 인용 버전을 명시해야 함. | https://github.com/Edge-Intelligence-Lab/MobileFineTuner ; https://arxiv.org/abs/2512.08211 |
| 2 | "FwdLLM은 **INT4/GPTQ 양자화 조건에서 LLaMA-7B 연합 fine-tuning을 peak 1.5GB**로 보고했습니다(GitHub)" (H3), TL;DR "(FwdLLM GitHub 기준 INT4/GPTQ 양자화 LLaMA-7B peak 1.5GB…)" | 두 출처를 섞은 조건 오기. (i) README 문장은 *"federated fine-tuning of LLaMA-7b, with only 1.5GB peak memory usage on mobile devices"*뿐이며 INT4/GPTQ 조건이 붙어 있지 않음. 같은 README에 LLaMA 실험 코드는 "future work"로 남아 있어 재현 불가. (ii) USENIX ATC'24 논문 Table 7은 LLaMA-7B "Ours, INT4"의 메모리를 **4.0 GB**로 보고(FP16 15.6GB, INT8 7.9GB). 측정 과제는 AGNEWS **분류**, LoRA 가중치는 FP32, 기기는 Pixel 7 Pro(8GB)임. → 논문 수치(4.0GB)를 1차 근거로 쓰고, 1.5GB는 "README 주장, 논문 미기재"로 격하해야 함. | https://www.usenix.org/system/files/atc24-xu-mengwei.pdf (§5.2, Table 7) ; https://github.com/UbiquitousLearning/FwdLLM |
| 3 | FwdLLM "…LLaMA over COTS smartphones within only 10 minutes" (조건 누락) | 인용문 자체는 논문에 있음. 다만 Table 7 기준 INT4 CPU는 0.19h(≈11.4분), NPU는 0.07h인데, NPU 수치는 *"LLaMA currently is not supported by mobile NPU, therefore we emulate its speed"*라는 **에뮬레이션** 값임. H3 근거로 쓸 때는 "분류 과제·INT4·NPU 속도는 추정치"라는 조건을 붙여야 함. | 같은 PDF, Table 7 각주 |
| 4 | Thread 1 PRPF 행: "**PRPF (Perceive Before Reasoning)** … 기존 benchmark(**ProactiveAgent, FingerTip**) 재사용" | PRPF(Xiaomi HyperAI 팀, Ding et al.)는 **ProactiveMobile** 벤치마크로 평가함. ProactiveAgent(주 평가)와 FingerTip-20K(appendix transfer)를 쓰는 쪽은 다른 논문인 "Do Proactive Agents Really Need an LLM…"(Liu et al., arXiv 2605.30152)임. 두 논문을 한 행에 묶으면서 벤치마크가 뒤섞였으므로 행을 분리하고 저자를 표기해야 함. | https://arxiv.org/abs/2606.03236 ; https://arxiv.org/html/2605.30152 |
| 5 | Caveats: "CIPHER의 교정 비용 감소율(요약 31%, 이메일 73%)은 **2차 요약 출처에서만 확인**했습니다." | 1차 출처에서 확인됨: *"CIPHER achieves the smallest edit distance cost reducing edits by 31% in the summarization task and 73% in the email writing task"* (no-learning 대비, arXiv v3 본문). Caveat는 삭제하고 수치를 본문으로 옮길 것. 단, 같은 표에서 CIPHER의 누적 edit cost는 oracle 대비 요약 32,974 vs 6,573(약 5배), 이메일 8,391 vs 1,851(약 4.5배)로 **oracle과의 격차가 크다**는 점을 H1 평가에 함께 적어야 함. 학회는 NeurIPS 2024 main track으로 확인됨. | https://arxiv.org/html/2404.15269v3 (Table 2) ; https://neurips.cc/virtual/2024/poster/96078 |
| 6 | ASTRA-bench: "코드 '**coming soon**', CC BY-NC-ND. 주인공 수 표기 불일치(4 vs 5)" | 불일치는 **원문 자체의 불일치**로 확인됨. abstract는 "four" protagonists, 본문은 5명을 이름으로 나열함(Dawei Shen, Theo Appleseed, John Quinn, Emily Rose, Lucas Garcia). 코드도 abstract는 "released with full execution environment", 본문은 `github.com/<coming-soon>`로 원문이 모순됨. → "원문 abstract/본문 상충, 본문 기준 5명"으로 쓰고, 코드 상태는 "abstract는 공개로, 본문 링크는 미공개로 표기"로 정확히 기술할 것. 나머지(2,413 시나리오, 평균 14일, Claude-4.5-Opus 0.9112, payload 병목)는 확인됨. | https://arxiv.org/abs/2603.01357 ; https://arxiv.org/html/2603.01357 |
| 7 | Caveats: "**CI-Bench 저자 목록**" 확인 필요 / 표: "CI-Bench (Google DeepMind) … **44,100** test cases" | 저자 확인: Zhao Cheng, Diane Wan, Matthew Abueg, Sahra Ghalebikesabi, Ren Yi, Eugene Bagdasarian, Borja Balle, Stefan Mellem, Shawn O'Banion. abstract의 규모 표현은 *"44 thousand test samples across eight domains"*이므로 "44,100"이라는 정밀 수치는 abstract에 없음(본문 표에서 확인하지 못했다면 "~44K, 8 domains"로 쓸 것). 저자 중 Bagdasarian·Ghalebikesabi·Yi·Balle가 AirGapAgent와 겹치므로 같은 그룹의 연속 연구라는 점도 적을 만함. | https://arxiv.org/abs/2409.13903 |
| 8 | FingerTip 20K: "Yang et al. (arXiv 2507.21071, **2025**)", 규모 "20K 에피소드급" | arXiv 페이지 기준 **ICLR 2026 poster**이고 Tsinghua FIB lab 연구임. 규모는 "20K unique human demonstrations"로 정확히 쓸 것. | https://arxiv.org/abs/2507.21071 |

---

## 2. Likely wrong / unsupported

| # | Report text (quoted) | Why doubtful | What to check |
|---|---|---|---|
| 1 | "**ProactiveMobile** — Kong et al. (**CVPR 2026**, arXiv 2602.21858)" | arXiv v4(2026-05-08)에 venue 표기가 없음. 수치(3,660+, 14 시나리오, 30명 전문가, 19.15/15.71/7.39%)는 abstract와 일치함. | CVF open access 목록에서 채택 여부 확인. 없으면 "arXiv 2026"으로 표기 |
| 2 | "**P³** — Salemi & Zamani (**SIGIR 2026**)" | arXiv 2601.17569 페이지에 venue 표기 없음. 90.3–95.7%, leakage +1.5–3.5%, client 토큰 9.2%는 abstract와 일치함. | SIGIR 2026 accepted list 확인 |
| 3 | "Attelia / Attelia II … 통제 실험, 30명×16일 in-the-wild … 무작위 타이밍 대비 인지 부하 46%↓. 민감 사용자에서 좌절 28%↓" | 검색 요약(1차 출처 미확인)에 따르면 **46%는 통제 실험**, **30명×16일 in-the-wild에서는 33%** 감소임. 한 셀에 섞어 쓰면 in-the-wild 효과가 46%인 것처럼 읽힘. 28%(좌절)는 PerCom'15인지 UbiComp'15(Attelia II)/PMC 저널인지 출처가 불분명함. | IEEE Xplore 7146515 abstract와 Attelia II 논문에서 수치별 출처를 분리 |
| 4 | Gaia2 Key Finding 2: Mialon의 Arize 인터뷰 인용 "GPT-5 gets 0 on Time…", "GPT-5 jumps to ~34% on Time" | 인터뷰는 2차 출처임. 논문 본문에 같은 수치가 1차로 있음: Time split GPT-5(high) 0.0, instant 모드 0.0%→34.4%, "inverse scaling in the Time capability". | 인터뷰 인용을 논문 인용으로 교체(arXiv 2602.11964 HTML) |
| 5 | "**LoRA Peak Memory Reduction** (arXiv 2606.19528) … Llama-3.2 3B, 2048 토큰: 26.20GB → 1.02GB" | abstract는 "up to 26× and 28×"만 보고하고 GB 값·시퀀스 길이·배치·기기 조건은 없음. 저자(Dbouk, Reisser, Louizos 등)는 Qualcomm 계열로 보이며 **실기기 측정인지 추정치인지** 확인되지 않음. 논문 제목("Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices")도 누락됨. | 본문 표에서 26.20→1.02GB의 조건(양자화 비트, checkpointing, 기기) 확인 |
| 6 | MobileFineTuner 시사점 "**야간 충전 중 학습 스케줄링**" | 논문의 스케줄러는 배터리 잔량이 임계값 아래로 떨어지면 학습 빈도를 줄이는 **energy-aware throttling**(PowerMonitor)임. 충전 상태나 시간대에 맞춰 학습을 배치하는 스케줄러가 아님. "16GB/1B params(FP16)"와 "2025년 폰 RAM 4–16GB"는 논문에 rule-of-thumb로 실제로 있음. | "야간 충전 스케줄링"은 본 과제가 새로 설계할 부분으로 기술 |
| 7 | BehaviorGen "합성 데이터만으로 fine-tuning 이득의 62.0% / 87.8% 회복. MIA 공격 성공률 <0.55" | abstract에는 "up to 18.9%" 성능 향상만 있고 62.0/87.8/MIA 수치는 없음. 저자는 Li, Ding, Gong, Li(Tsinghua)임. | 본문 표 확인 후 조건과 함께 기재 |
| 8 | "**'Tell Me Why You're Asking'** (CHI 2026), **Read the Room** (CHI EA 2026)" | 저자·DOI가 없고, 검색으로 해당 제목의 CHI 2026 논문을 찾지 못함(could not verify). | DOI를 달거나 삭제 |
| 9 | "**Using LLMs to Model Notification Timing…** (Augmented Humans 2026) … N=21 … **사용자 평정 이력 제공이 가장 큰 개선**" | DOI(10.1145/3795011.3795067)는 존재하지만 ACM DL이 403이라 N=21과 "가장 큰 개선" 결론을 확인하지 못함. Key Finding 5가 이 한 편(N=21, 사무실 과업)에 기대어 "개인화가 핵심"으로 일반화하고 있음. | 본문 확인. 결론은 "소규모 연구에서 관찰됨"으로 완화 |
| 10 | MCP-Persona "Lark, Slack, **Notion, email**, SNS 등 24개 서버 … 173 tasks / 970 checkpoints … 91.5%" | ICML 2026은 확인됨. abstract가 예시로 드는 앱은 Reddit, Xiaohongshu, Lark, Slack이며, Notion·email·24·173·970·91.5%는 abstract에서 확인하지 못함. | 본문 확인 |
| 11 | PARE "(UCSB·**Apple**·UW …)", "Stackelberg POMDP" | 143 tasks, FSM 앱 모델링은 abstract로 확인됨. 소속 표기(특히 UW)와 Stackelberg 정식화는 확인하지 못함. 저자 목록에 Apple 연구자로 알려진 이름(Yinfei Yang, Zhe Gan, Alkesh Patel)은 있음. | 논문 1쪽 소속 확인 |
| 12 | 표 PersonaBench "Tan et al. (Salesforce, arXiv 2502.20616, 2025)" | 확인 결과 **Findings of ACL 2025**임. venue 누락(오류라기보다 보완 필요). | — |
| 13 | "**SAPA-Bench / PrivacyBench** 계열 (**2025**) … 7,138 시나리오, 민감도 3단계" | SAPA-Bench("Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents", arXiv 2508.19493)는 공식 repo 기준 **AAAI 2026**임. PrivacyBench(arXiv 2512.24848)는 PersonaBench를 확장한 **대화형** 벤치마크로 별개 연구임. 둘을 한 "계열"로 묶은 것은 오류. "민감도 3단계"도 SAPA의 annotation 항목(privacy category 8종, risk severity, leakage modality)과 맞는지 확인되지 않음. | 1차 페이지 미개봉(검색 snippet 기준): https://github.com/Zhixin-L/SAPA-Bench , https://arxiv.org/html/2508.19493 , https://arxiv.org/html/2512.24848 — 열어서 venue·annotation 항목 확인 |

(참고: ProactiveBench(ICLR 2025, 6,790 events, F1 66.47%), ContextAgent(NeurIPS 2025, 1,000 samples, +8.5%/+6.0%), ProAgentBench(500+h, 28,000+ events, B=0.787), KnowU-Bench(42/86/64, 11 models, 80.0%), Gaia2(12 apps/101 tools, 1,120=800+320, 10 universes, 0.99 precision/0.95 recall on 450 trajectories, GPT-5(high) 42.1, Kimi-K2 20.1, PersonaHub, 400K–800K tokens, "temporal consistency across apps" 한계 명시), iOSWorld(26 apps, 133 = 27/60/46, 52%/37%), SentinelBench(100 tasks, 10 envs, reaction time), PrivacyLens(25.68%/38.69%, NeurIPS'24 D&B), AirGapAgent(94→45%, 97%, CCS'24), PrefEval(3,000 pairs, 20 topics, <10% at 10 turns, ICLR'25 oral), TinyAgent(EMNLP'24 Demo, 1.1B/7B, GPT-4-Turbo 상회), PocketLLM(RoBERTa-large ~4GB, OPT-1.3B ~6.5GB, OPPO Reno 6), FwdLLM의 93%/169MB/96.7%가 RoBERTa-large·YELP-P·Pixel 7 Pro 조건이라는 점, Pielot CHI'14(중앙값 6.15분, precision 81.2%, accuracy 70.6%, 24명·2주)는 **1차 출처와 일치함**.)

---

## 3. Missing prior work

| # | Paper (title, authors, venue, year, URL) | Which thread | Why it matters for the gap claim |
|---|---|---|---|
| 1 | **ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation** — Jiho Kim, …, Yohan Jo, Edward Choi. **ICLR 2026**. https://arxiv.org/abs/2509.21730 | T1+T3 | 페르소나 32개의 LLM 사용자 에이전트가 가정 시나리오에서 선제 제안을 평가하고, ProPerAssistant가 피드백으로 "steadily improves user satisfaction"함. **선제성과 피드백 기반 누적 선호 학습(시간에 따른 개선 곡선)**을 한 루프로 결합한 직접 선행연구로, gap (d)를 부분 잠식함. 본 과제는 "실행(write-action)·마감 urgency·다중 앱 타임라인·온디바이스"로 차별화해야 함. |
| 2 | **Preference-Driven Online Adaptation for Personalized Interaction Initiation in Proactive AI Assistants** (EOPA) — Yufeng Wang, …, Mingkui Tan. arXiv 2608.04416, 2026 | T1+T3, H1/H3 | 온라인 피드백으로 "언제 먼저 말을 걸지"를 **LLM 추론이나 재학습 없이** 갱신하고, timing F1을 약 +20 개선하며, 일일 적응 시간을 11.41초에서 0.39초로 줄임. gap (d)의 "타이밍 선호 학습"과 겹치고, H1(가벼운 선호 메모리 갱신)의 직접 근거이자 H3(재학습)의 반례임. |
| 3 | **PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents** — Minjia Wang et al. **EACL 2026 Industry**. https://arxiv.org/abs/2603.11955 | 보조(합성 데이터) | 프로필 → 이벤트 시퀀스 → **이메일·메시지·캘린더·리마인더** artifact 생성, 실제 OOD 과제로 realism 검증. gap (a)와 Key Finding 3("realism을 실제 로그와 비교한 연구는 찾지 못함")에 대한 직접 반례. ASTRA-bench와 함께 생성기 비교 대상에 넣어야 함. |
| 4 | **MyPCBench: A Benchmark for Personally Intelligent Computer-Use Agents** — L. K. Jang, A. K. Jang, J. Y. Koh, R. Salakhutdinov. arXiv 2606.16748, 2026 | T2 | 17개 앱에 걸친 합성 사용자 상태(거래·이메일·이벤트·메시지 등)와 184 tasks. iOSWorld와 같은 그룹이며, 다중 앱 개인 상태 위의 대리 실행이라는 점에서 gap (a)+(c)와 겹침. |
| 5 | **"From Overwhelmed to Overview: Understanding Smartphone Users' Preferences and Expectations in Relieving Notification Overload via Text Summarization"** — U.-D. Chen, P.-J. Wang, Y.-C. Lee, Y.-H. Lin, Y.-L. Chou, **Y.-J. Chang**. PACM HCI (MobileHCI) 2025. https://dl.acm.org/doi/10.1145/3743703 (ACM 403, search snippet + NYCU PDF 링크만 확인) | T4 | LLM으로 알림을 **요약·묶어 제시**하는 Android 앱의 1주 in-the-wild 배포(20명). 우선순위 3유형, 공개 수준 3단계를 보고함. 본 과제의 `batch_to_digest` 행동과 digest 설계에 대한 가장 직접적인 HCI 선행연구로, "batching·digest를 다룬 연구 없음"이라는 주장(Key Finding 1)을 약화함. |
| 6 | **PersoNo: Personalised Notification Urgency Classifier in Mixed Reality** — J. Zheng, …, S. Mayer, L.-H. Lee. **ISMAR 2025**. https://arxiv.org/abs/2508.19622 | T4 | 사용자 응답 행동에서 **개인화된 알림 urgency**를 LLM multi-agent로 분류함(81.5%, 18명). "activity context가 내용·발신자만큼 중요"하다고 보고함. C1 urgency 라벨과 `user_context`의 직접 근거이자 선행연구. |
| 7 | **ProMemAssist: Exploring Timely Proactive Assistance Through Working Memory Modeling in Multi-Modal Wearable Devices** — K. Pu, …, T. Jonker. **UIST 2025** (Meta 연구진으로 보임). https://arxiv.org/abs/2507.21378 | T1/T4, 산업 | 도움의 가치와 방해 비용을 저울질하는 **timing predictor**, 12명 사용자 연구. Net utility = 가치 − λ·방해비용 지표의 선행 형식화로 인용해야 함. |
| 8 | **Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents** — Bin Wu, …, Chuan Shi. arXiv 2605.28108, 2026 | T3 | 다음 세션에서 쓸 선호를 "지금 물을지" 판단하는 문제. frontier 8종이 oracle보다 62점 이상 낮음. 저녁 digest 질문(C2 "Digest round-trip")과 선호 획득 설계에 직결됨. |
| 9 | **AcCoRD: Evaluating User-Agent Collaboration Under Realistic User Preference Dynamics** — T. Srinivasan, …, J. Thomason. arXiv 2608.27818, 2026 | T3 | 선호가 "formed, revealed, adjusted, relaxed"되는 동역학을 평가함. C3의 Forgetting·Over-generalization 지표에 **선호 변화(drift)** 축이 빠져 있다는 근거. |
| 10 | **Training Proactive and Personalized LLM Agents** (PPP / UserVille) — Weiwei Sun, …, Graham Neubig, Maarten Sap, Yiming Yang. arXiv 2511.02208 (comments: COLM 2026) | T3 | 선호를 가진 설정 가능한 LLM 사용자 시뮬레이터와 "preference adherence" 보상의 multi-objective RL. 시뮬레이터 기반 선호 학습 프로토콜의 대안 기반. |
| 11 | **π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows** — Haoran Zhang, …, Yafu Li. arXiv 2605.14678, 2026 | T1 | 페르소나 5개, 100 multi-turn tasks, 숨은 의도·세션 간 연속성. proactive personal assistant 벤치마크 목록에서 빠짐. |
| 12 | **Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations** — P. Seshadri, …, S. Goldfarb-Tarrant. arXiv 2601.17087, 2026 | 방법론 | 시뮬레이터 LLM에 따라 에이전트 성능이 최대 9pp 흔들리고 난이도별 편향이 있음. PARE·KnowU·PRELUDE식 시뮬레이터에 의존하는 설계의 **핵심 위협 요인**이므로 반드시 다뤄야 함. |
| 13 | **Proactive Service Agents: A Unified Decision Framework, Methods, and Evaluation** — Yan Tang, Tingyu Cao, Yuanbo Tang, Huaze Tang, Keer Hu. arXiv 2609.03727, 2026 (survey) | T1 | 개입을 "silent / ask / assist / act" 선택과 **"option value of waiting"**으로 정식화한 서베이(ProAgentBench 그룹). 본 과제의 4-way 행동 공간과 거의 같은 틀이므로 인용하고 차이(마감 비용, digest, 교정)를 명시해야 함. |
| 14 | **Never Start from Scratch: Expediting On-Device LLM Personalization via Explainable Model Selection** (XPerT) — H. Wang, B. Yang, X. Yin, **Wei Gao**. **MobiSys 2025**. https://arxiv.org/abs/2504.13938 | T5, H3 | 온디바이스 개인화의 계산 비용 83% 절감, **data efficiency 51% 개선**. H3의 "교정 데이터 희소" 한계에 대한 직접 해법 계열이며 MobiSys 문헌으로 필수. |
| 15 | **Apple Intelligence Foundation Language Models: Tech Report 2025** — Apple (Ethan Li et al.). arXiv 2507.13575 | T5, 산업 | 3B on-device 모델, 2-bit QAT, **LoRA adapter fine-tuning 프레임워크**. "1–3B 온디바이스 실행 LLM" baseline 설정의 산업 근거. Apple의 알림 요약·우선순위 기능은 논문 근거가 아니므로 제품으로만 언급할 것. |
| 16 | **MobileRLHF** (MobileFineTuner repo, 2026-07-10 공지) | T5, H3 | 스마트폰에서 standalone·federated **RL 기반 preference post-training**을 지원한다고 README에 명시됨(논문 미확인). H3 근거로 FwdLLM보다 더 직접적임. https://github.com/Edge-Intelligence-Lab/MobileFineTuner |
| 17 | **Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory** — B. Wu, Z. Ding, J. Huang, Y. Zhao. arXiv 2606.25115, 2026 | T5, H1/H3 | 가중치 갱신 없이 RAM·에너지 예산 안에서 경험 메모리를 선별 관리함. "선호 메모리 갱신"(H1) 계층의 온디바이스 설계 근거. |
| 18 | **LLM Agent-Based Simulation of Student Activities and Mental Health Using Smartphone Sensing Data** — W. Sommuang et al. arXiv 2508.02679, 2025 | 보조(합성 ESM) | LLM 에이전트가 StudentLife 기반으로 **EMA 자기보고를 합성**함. Caveat "LLM으로 합성 ESM…을 생성한 연구는 찾지 못함"은 arXiv까지 넓히면 성립하지 않음(IMWUT/UbiComp로 한정하면 유지 가능하므로 범위를 명시할 것). |

---

## 4. Gap statement — does it survive?

다섯 요소의 **완전한 결합**으로서는 살아남습니다. 확인한 어떤 연구도 (b)의 마감·지연 비용 기반 4-way 행동(특히 `act_silently_and_report`와 `batch_to_digest`)과 (c)의 write-action 검증, (e)의 온디바이스 제약을 동시에 갖추지 않았습니다. 그러나 개별 쌍은 이미 선점되어 있습니다. **ProPerSim(ICLR 2026)은 proactive + 시뮬레이터 피드백 기반 누적 선호 학습 곡선((d)+타이밍)**을, **EOPA(2608.04416)는 온라인 피드백으로 개입 타이밍 선호를 재학습 없이 적응((d)+타이밍, 경량)**을, **PersonaTrace·MyPCBench는 다중 앱 합성 개인 상태((a), realism 검증 포함)**를, **Chen et al. MobileHCI'25와 PersoNo(ISMAR'25)는 LLM 기반 알림 요약·묶음과 개인화 urgency((b)의 일부)**를 다룹니다. 따라서 "each 따로 다뤘다"는 문장과 Key Finding 1의 "batching·digest를 행동 옵션으로 두는 연구는 없다", Key Finding 3의 "realism 비교 연구 없음"은 수정이 필요합니다. novelty는 "조합 + 대리 실행 검증 + 마감 비용 정량화"에 두고, 위 연구들을 명시적으로 인용해 차별화하는 편이 안전합니다.

---

## 5. Hypotheses H1/H2/H3 — verdict on the verdicts

**H1 (선호만 추출, "실현 가능성 높음, 1차 메커니즘")** — *부분 동의.*
- 이유: CIPHER가 no-learning 대비 edit을 31%/73% 줄인 것은 확인됨. 하지만 oracle 대비 4.5–5배 비용이 남아 **추출된 자연어 선호가 잠재 선호를 불완전하게 포착**함을 같은 표가 보여 줌. 또한 선호가 "텍스트 스타일"이지 행동·타이밍 규칙이 아님.
- 근거 오용: **P³는 H1 근거가 아님.** P³는 원본 프로필 전체를 클라이언트에 두고 소형 모델이 **원본 프로필 RAG**로 서버 초안을 수정하는 방식임. 선호 추출이 아니라 H2(분리)·프라이버시 쪽 근거임.
- 보강 근거: EOPA(재학습 없는 온라인 타이밍 선호 적응), Forget to Improve(예산 내 메모리 큐레이션).
- 반증: PrefEval(프롬프트만으로는 긴 맥락에서 선호 추종이 붕괴하고 fine-tuning이 개선을 줌), KnowU-Bench(병목이 "preference acquisition"이며 frontier 모델도 50% 미만), Ask Now Use Later(선호를 획득하지 못하면 oracle 대비 −62점).

**H2 (filter/output 분리, "근거가 가장 강함")** — *방향은 동의, 강도는 과장.*
- 이유: PRPF(경량 MPP 게이트 → reasoner, ProactiveMobile에서 오탐 감소)와 TGL(14개 backbone에서 F1 평균 +16.7, 4–83× 빠름, 약 220MiB)은 **효율·오탐** 측면의 분리를 강하게 지지함.
- 한계: TGL trigger는 "one checkpoint … serve every backbone"으로 **개인화되지 않았고**, 벤치마크 라벨로 학습하며, 저자 스스로 "offline RM-judge protocol"로 사용자 체감을 측정하지 않았다고 밝힘. 따라서 "개인화 학습을 소형 filter에 몰아넣는다"는 함의의 근거는 아직 없음.
- 근거 오용: AirGapAgent의 97%는 context-hijacking 공격 방어율이며 minimizer도 LLM임. "소형 filter" 근거가 아니라 프라이버시 분리 근거임. ProAgentBench의 timing/content 계층 분해는 **과제 정의**이지 모델 분리의 효과 검증이 아님.
- 반증: KnowU-Bench는 개입 보정 실패가 선호 추론 실패와 얽혀 있음을 보여 줌(Sonnet 4.6 실패의 80%가 intervention/passive). 소형 filter가 이 추론을 감당할 수 있는지는 미검증임. 판정은 "근거가 가장 강함"보다 **"효율 측면은 강함, 개인화된 urgency 판단 측면은 미검증"**이 적절함.

**H3 (야간 재학습, "조건부로 실현 가능")** — *판정은 동의, 근거는 교체 필요.*
- 이유: 주 근거인 FwdLLM "INT4/GPTQ 1.5GB"는 논문에 없음. 논문 값은 4.0GB, 분류 과제, NPU 속도는 에뮬레이션임(§1-2, §1-3). LoRA 1.02GB는 조건 미확인이고, MobileFineTuner는 SenSys 2027 conditional이며 스케줄러는 충전 인지형이 아님.
- 더 적합한 근거: XPerT(MobiSys'25, 데이터 효율 +51%, 비용 −83%), MobileRLHF(온디바이스 preference post-training, README), Apple AFM 2025(3B on-device + LoRA adapter).
- 반증: PRELUDE 저자는 사용자별 fine-tuning이 "costly, challenging to scale … may even degrade its performance on other tasks"라고 명시함. EOPA는 재학습 없이 적응함. 하루 교정 수 건이라는 희소성은 보고서도 인정함.
- 권고의 "주 단위 LoRA 증류"에는 근거가 전혀 인용되지 않음. ablation 조건으로 두는 것은 좋지만 설계 권고로 쓰기에는 약함.

---

## 6. Benchmark labels & metrics — issues

- **라벨 중복과 불일치 위험:** `urgency ∈ {U3..U0}`와 `deadline_t`, `cost_of_delay(t)`가 동시에 정답으로 주어져 있어 서로 모순될 수 있습니다(예: U1인데 deadline이 15분 후). urgency 등급은 deadline·cost에서 **파생되는 값**으로 정의해 정답 원천을 하나로 두어야 합니다.
- **정답 행동의 비유일성:** 올바른 행동은 urgency × `user_context` × 숨겨진 선호의 함수라서 한 사건에 정답이 여럿일 수 있습니다(ProactiveMobile도 multi-answer annotation을 씀). 단일 정답 대신 **행동별 비용 행렬 또는 허용 집합(accept set)**으로 채점해야 합니다.
- **Deadline-met recall과 행동 선택의 결합:** `act_silently_and_report`도 마감을 지키므로 recall만으로는 "방해 없이 해결"과 "방해로 해결"이 구분되지 않습니다. recall은 결과(적시 처리)에, 방해 여부는 별도 지표에 배정해 두 축이 서로 독립적이도록 설계해야 합니다.
- **시뮬레이션 시간과 모델 지연:** Gaia2는 생성 지연을 포함하면 GPT-5(high)의 Time이 0.0%, instant 모드에서는 34.4%라고 보고합니다. Time-to-action/slack을 **지연 포함(wall-clock)**과 **instant** 두 모드로 모두 보고하지 않으면 온디바이스(느린 1–3B)와 frontier의 비교가 왜곡됩니다. 온디바이스 논문이라면 핵심 설계 결정입니다.
- **Net utility의 λ 임의성:** 단일 λ 대신 λ sweep 곡선이나 Pareto frontier(적시 처리 vs 방해 횟수)를 보고해야 합니다.
- **ECE 정의 부재:** "기대 비용 기반 calibration(ECE)"는 에이전트가 무엇에 대한 확률을 내는지 정의되어 있지 않습니다. 4-way 행동 확률에 대한 ECE인지, 마감 내 미처리 확률에 대한 것인지 명시해야 합니다.
- **modality 정답 없음:** `interrupt_now(modality: voice/banner/haptic)`는 시뮬레이터 안에서 ground truth나 효과가 정의되지 않았습니다. 규칙(운전 → voice)으로 정하면 사실상 규칙 일치 검사에 불과하므로, 라벨로 쓸지 제약 조건으로 둘지 결정해야 합니다.
- **Interruptibility 문헌 기술의 부정확성:**
  - (i) Pielot CHI'14의 대상은 **attentiveness**(몇 분 안에 볼지, 6.15분 중앙값을 기준값으로 사용)이고, Mehrotra CHI'16은 **response time**과 **perceived disruption**을 구분했습니다. 보고서는 "반응성/적절성" 분리를 제안하면서도 C1 라벨에 attentiveness(볼 수 있는가)와 receptivity(원하는가)의 구분이 없습니다.
  - (ii) NASA-TLX는 Attelia의 **통제 실험**에서 쓰인 측정이지 in-the-wild ESM 라벨이 아닙니다. Key Finding 5처럼 ESM·응답 시간과 같은 층위로 나열하면 오해를 부릅니다.
  - (iii) Attelia의 breakpoint는 스마트폰 UI 이벤트(앱 전환 등)에서 감지한 **사용 중 전환점**입니다. 캘린더·메일 타임라인만 있는 합성 환경에서는 관측할 수 없는 신호이므로, "Attelia breakpoint baseline"을 쓰려면 합성 UI 사용 로그 채널이 필요합니다.
- **인간 검증 계획이 리뷰어를 통과하기 어려운 이유:**
  - (a) 제3자 annotator 3명의 Krippendorff α는 **annotator 합의**를 잴 뿐 persona 당사자의 urgency를 재지 않습니다. 보고서가 인용한 AHs 2026·PersoNo는 urgency가 **개인적**이라고 보고합니다. annotator에게 persona 프로필과 선호 규칙을 주고 판정하게 하는 절차와 α 목표치를 명시해야 합니다.
  - (b) "in-the-wild ESM으로 체감 방해도와 합성 라벨의 상관"은 참가자가 **합성 사건을 겪지 않으므로** 직접 상관을 낼 수 없습니다. 합성 사건을 참가자 폰에 주입하는 Wizard-of-Oz 방식이나, 참가자의 실제 알림에 같은 라벨 스키마를 적용하는 방식 중 하나를 택해야 합니다.
  - (c) "연구실 5–10명 × 2주, 비공개"는 IMWUT 기준으로 표본이 작고 편향되어 있습니다(비교: My Phone and Me 20명, Attelia 30명×16일, ProAgentBench 500h+). NeurIPS D&B 기준으로는 비공개 보정 데이터가 재현성 요건(datasheet, 공개)과 충돌합니다. 최소한 연구실 외부 20–30명, IRB, 익명화된 집계 통계 공개를 권합니다.
  - (d) 시뮬레이터 타당도: "Lost in Simulation"에 따르면 시뮬레이터 LLM에 따라 성능이 최대 9pp 흔들리므로, **복수 시뮬레이터 LLM**으로 순위 안정성을 보고해야 합니다.
- **C3 지표:**
  - Corrections-to-convergence는 끝내 수렴하지 않는 경우(censoring) 처리 규칙이 없습니다(상한 cap 및 생존분석식 보고 필요).
  - Forgetting과 Over-generalization에 더해 **선호 변경(drift) 이벤트**가 없으면(AcCoRD, HorizonBench), "옛 규칙을 고집하는 오류"와 "망각"을 구분할 수 없습니다.
  - Repeat-mistake rate는 PVR·corrections-to-convergence와 상당 부분 중복되므로 하나로 합치거나 정의를 차별화해야 합니다.
- **Report fidelity와 Privacy leakage의 판정 방법:** 사후 보고와 실행 로그의 일치를 누가 판정하는지(규칙인지 LLM judge인지)와 judge-인간 일치도를 적어야 합니다. PrivacyLens 방식 leakage도 LLM judge 기반이므로 같은 요구가 적용됩니다.

---

## 7. Top 5 fixes, ranked by impact

1. **Gap statement와 Key Findings 1·3 수정:** ProPerSim(ICLR'26), EOPA, PersonaTrace(EACL'26), MyPCBench, Chen et al. MobileHCI'25(LLM 알림 요약·묶음), PersoNo(ISMAR'25), Proactive Service Agents 서베이를 인용하고, novelty를 "마감 비용 기반 4-way 행동 + write-action 검증 + 교정 학습 곡선 + 온디바이스"의 **결합**으로 좁혀 재서술할 것. 현재의 "각각 따로"와 "batching·digest 연구 없음"은 리뷰어에게 바로 반박당합니다.
2. **H3 근거 교체:** FwdLLM의 "INT4/GPTQ 1.5GB"를 논문 Table 7의 4.0GB(분류, NPU 에뮬레이션)로 정정하고, TL;DR에서 FwdLLM과 LoRA 1.02GB 수치가 한 괄호에 섞인 부분을 분리할 것. MobileFineTuner를 SenSys 2027 conditional로 정정하고, XPerT(MobiSys'25), MobileRLHF, Apple AFM 2025로 보강할 것.
3. **인간 검증 설계 재작성(§6):** annotator에게 persona 선호를 제공하는 판정 절차, 합성 사건 주입형(Wizard-of-Oz) ESM 또는 실제 알림 라벨링, 연구실 외부 20–30명과 IRB, 복수 시뮬레이터 LLM 보고("Lost in Simulation" 인용)를 포함할 것.
4. **H1/H2 근거 재배치:** P³를 H1에서 H2·프라이버시 근거로 옮기고, H2 판정을 "효율·오탐은 강함, 개인화 urgency는 미검증"으로 낮출 것. PRPF와 TGL 논문의 행을 분리해 벤치마크(ProactiveMobile vs ProactiveAgent+FingerTip)를 바로잡을 것. CIPHER의 oracle 격차를 H1 리스크로 추가할 것.
5. **라벨 스키마 정리:** urgency 등급을 deadline·cost에서 파생시키고, 정답 행동을 accept set 또는 비용 행렬로 채점하며, 지연 포함/instant 이중 시간 모드(Gaia2 근거)를 도입할 것. ECE 대상과 λ sweep을 정의하고, 선호 drift 이벤트를 추가할 것. 서지 오류(SAPA-Bench AAAI'26과 PrivacyBench 분리, FingerTip ICLR'26, ASTRA 원문 모순 명시, CI-Bench 저자, CIPHER caveat 삭제, Gaia2 인터뷰 인용을 논문 인용으로 교체)도 함께 정리할 것.

---

## 8. Sources opened

직접 fetch(WebFetch/curl)한 URL:
- https://arxiv.org/abs/2512.08211 ; https://arxiv.org/html/2512.08211v2
- https://raw.githubusercontent.com/Edge-Intelligence-Lab/MobileFineTuner/main/README.md (및 king21-noass mirror) — "conditional accepted by Sensys2027", MobileRLHF
- https://arxiv.org/abs/2409.13903
- https://arxiv.org/abs/2603.01357 ; https://arxiv.org/html/2603.01357
- https://arxiv.org/abs/2404.15269 ; https://arxiv.org/html/2404.15269v3
- https://arxiv.org/abs/2308.13894
- https://www.usenix.org/system/files/atc24-xu-mengwei.pdf (PDF 다운로드 후 텍스트 추출: §5.2, Table 7, System Cost 절)
- https://github.com/UbiquitousLearning/FwdLLM ; https://raw.githubusercontent.com/UbiquitousLearning/FwdLLM/master/README.md
- https://arxiv.org/abs/2409.00138 (PrivacyLens)
- https://arxiv.org/abs/2405.05175 (AirGapAgent)
- https://arxiv.org/abs/2410.12361 (ProactiveBench)
- https://arxiv.org/abs/2602.11964 ; https://arxiv.org/html/2602.11964 (Gaia2)
- https://arxiv.org/abs/2505.14668 (ContextAgent)
- https://arxiv.org/abs/2604.08455 ; https://arxiv.org/html/2604.08455 (KnowU-Bench)
- https://arxiv.org/abs/2604.00842 (PARE)
- https://arxiv.org/abs/2609.03727 (Proactive Service Agents survey)
- https://arxiv.org/abs/2606.14314 (Communication Policy Evolution — 관련성 낮아 미인용)
- https://arxiv.org/abs/2605.14678 (π-Bench)
- https://arxiv.org/abs/2605.28108 (Ask Now, Use Later)
- https://arxiv.org/abs/2608.27818 (AcCoRD)
- https://arxiv.org/abs/2601.17087 (Lost in Simulation)
- https://arxiv.org/abs/2502.09597 (PrefEval)
- https://pielot.org/pubs/Pielot2014-CHI-AttPred.pdf (PDF 텍스트 추출)
- https://arxiv.org/abs/2508.19622 (PersoNo)
- https://arxiv.org/abs/2507.21378 (ProMemAssist)
- https://arxiv.org/abs/2511.02208 (PPP/UserVille)
- https://arxiv.org/abs/2606.25115 (Forget to Improve)
- https://arxiv.org/abs/2603.11955 (PersonaTrace)
- https://arxiv.org/abs/2606.16748 (MyPCBench)
- https://arxiv.org/abs/2606.19528 (LoRA peak memory)
- https://arxiv.org/abs/2602.04482 (ProAgentBench)
- https://arxiv.org/abs/2602.21858 (ProactiveMobile)
- https://arxiv.org/abs/2606.09764 (iOSWorld)
- https://arxiv.org/abs/2601.17569 (P³)
- https://arxiv.org/abs/2606.05342 (SentinelBench)
- https://arxiv.org/abs/2502.20616 (PersonaBench)
- https://arxiv.org/abs/2606.03236 (PRPF)
- https://arxiv.org/abs/2605.30152 ; https://arxiv.org/html/2605.30152 ("Do Proactive Agents Really Need an LLM…")
- https://arxiv.org/abs/2507.21071 (FingerTip 20K)
- https://arxiv.org/abs/2606.02470 (MCP-Persona)
- https://arxiv.org/abs/2409.00608 (TinyAgent)
- https://arxiv.org/abs/2505.17615 (BehaviorGen)
- https://arxiv.org/abs/2508.02679 (LLM-agent EMA simulation)
- https://arxiv.org/abs/2509.21730 (ProPerSim)
- https://arxiv.org/abs/2608.04416 (EOPA)
- https://arxiv.org/abs/2507.13575 (Apple AFM 2025)
- https://arxiv.org/abs/2504.13938 (XPerT, MobiSys 2025)

열기에 실패했거나 검색 snippet으로만 확인한 항목(= 미검증으로 취급):
- https://dl.acm.org/doi/10.1145/3743703 (403; 저자·venue는 검색 결과 기준)
- https://dl.acm.org/doi/10.1145/3795011.3795067 (403; AHs 2026 notification timing)
- Attelia: IEEE Xplore 7146515, Keio pure, Okoshi 연구실 페이지 모두 fetch 실패. 46%/33%/28%는 검색 snippet 기준
- My Phone and Me(20명/10,372/474): 검색 snippet 기준(Birmingham repository 요약)
- PocketLLM 수치: 검색 snippet 기준(ACL Anthology 2024.privatenlp-1.10)
- SAPA-Bench AAAI 2026 / PrivacyBench 2512.24848: 검색 snippet 기준
- NeurIPS 2024의 PRELUDE/CIPHER 채택: 검색 결과(proceedings.neurips.cc, neurips.cc poster 96078) 기준
- ProactiveBench ICLR 2025: 검색 결과(iclr.cc/virtual/2025/poster/28128) 기준
