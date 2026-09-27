# Review by Claude Fable

대상: `review/report.md` (온디바이스 선제적 개인 비서 벤치마크 문헌·데이터셋 리뷰). 검토일 2026-09-27.
방법: 표에 인용된 논문 대부분의 arXiv abstract/HTML, GitHub, 학회 accepted-paper 페이지를 직접 열어 대조했습니다. ACM DL은 403으로 열리지 않아 UbiComp/CHI 계열 일부 수치는 "could not verify"로 남겼습니다. 열어본 URL 전체 목록은 §8에 있습니다.

---

## 1. Verified wrong (opened the source; report contradicts it)

| # | Report text (quoted) | Correction | Source URL |
|---|---|---|---|
| 1 | Thread 5 FwdLLM 행: "LLaMA-7B는 INT4/GPTQ 양자화 조건에서 peak 1.5GB(GitHub UbiquitousLearning/FwdLLM)" 및 H3: "FwdLLM은 INT4/GPTQ 양자화 조건에서 LLaMA-7B 연합 fine-tuning을 peak 1.5GB로 보고했습니다(GitHub)" | 논문 본문(Table 7, Pixel 7 Pro)의 LLaMA-7B peak memory는 **FP16 15.6GB / INT8 7.9GB / INT4(GPTQ) 4.0GB**이며, "1.5GB"라는 숫자는 논문 어디에도 없습니다. 1.5GB는 GitHub README의 한 줄 문구("federated fine-tuning of LLaMA-7b, with only 1.5GB peak memory usage on mobile devices")일 뿐이고 README는 양자화 조건을 명시하지 않습니다. 즉 "INT4/GPTQ 조건에서 1.5GB"는 두 출처를 잘못 결합한 것입니다. "10분" 주장은 INT4 조건이 맞습니다("Through orchestration with quantization (INT4)… within only 10 minutes"). 또한 FwdLLM은 **연합학습(forward-gradient FL)** 설정이지 단일 기기 야간 개인화 학습이 아닙니다. | https://arxiv.org/html/2308.13894 ; https://github.com/UbiquitousLearning/FwdLLM |
| 2 | Thread 1 마지막 행: "**PRPF (Perceive Before Reasoning)** (arXiv 2606.03236, 2026), **"Do Proactive Agents Really Need an LLM…"** (arXiv 2605.30152, 2026) \| 기존 benchmark(ProactiveAgent, FingerTip) 재사용" | PRPF는 **ProactiveMobile** 벤치마크에서만 평가했습니다(ProactiveAgent·FingerTip 언급 없음). ProactiveAgent(desktop) + FingerTip-20K(mobile, Appendix A)를 쓰는 것은 TGL 논문(2605.30152)뿐입니다. 두 논문을 한 행에 묶으면서 벤치마크 정보가 섞였습니다. | https://arxiv.org/abs/2606.03236 ; https://arxiv.org/html/2605.30152 |
| 3 | Thread 2 PersonaBench 행: "582 문항, 15 characters" | 본문: "The test set includes **1515 characters**, each with corresponding questions and ground-truth answers." 582는 문항 유형 합(269 basic + 186 preference + 127 social)으로 맞습니다. 또한 venue가 비어 있는데 arXiv 페이지에 **Findings of ACL 2025**로 표기되어 있습니다. | https://arxiv.org/html/2502.20616 ; https://arxiv.org/abs/2502.20616 |
| 4 | Thread 4 Attelia 행: "Attelia / Attelia II — Okoshi et al. (PerCom 2015; UbiComp 2015) … 무작위 타이밍 대비 인지 부하 46%↓" | 46%는 **Attelia(PerCom 2015)의 실험실(laboratory) 결과**이며 in-the-wild 결과가 아닙니다. Attelia II(UbiComp 2015)의 in-the-wild 결과는 "UI-event-only 대비 체감 workload 감소가 **71.8%** 더 큼"으로 보고됩니다. 표는 두 논문의 수치를 하나의 in-the-wild 결과처럼 배치하고 있습니다. ("좌절 28%↓", "30명×16일"은 ACM DL 403으로 확인 불가 — §2 참고) | https://www.semanticscholar.org/paper/b69d4d0a57a7dfebd42168b85661a4b14bd29d12 (Attelia) ; UbiComp'15 DOI 10.1145/2750858.2807517 (abstract elided) |
| 5 | Thread 5 LoRA Peak Memory 행: "Llama-3.2 3B, 2048 토큰: 26.20GB → 1.02GB" → "3B 야간 LoRA가 수치상 가능"; H3 근거에도 동일 | 숫자 자체는 Table 1에 있으나 **조건이 누락**되었습니다: 26.20GB는 **FP32 baseline**(A100에서 프로파일링), 1.02GB는 **INT4 base weight + FP32 activation, batch 1** 조건이며 논문은 "12GB mobile device"에서 검증했다고 씁니다. FP32 서버 baseline 대비 26×를 "폰에서 26GB가 1GB로 줄었다"로 읽히게 쓰면 리뷰어가 지적합니다. | https://arxiv.org/html/2606.19528 |
| 6 | Caveats: "CIPHER의 교정 비용 감소율(요약 31%, 이메일 73%)은 2차 요약 출처에서만 확인했습니다." | 1차 출처(v3 본문)에 그대로 있습니다: "CIPHER achieves the smallest edit distance cost reducing edits by **31%** in the summarization task and **73%** in the email writing task" — 비교 대상은 최강 baseline ICL-edit, T=200 rounds, token-level Levenshtein 누적 edit distance. Caveat를 지우고 조건을 표에 적으면 됩니다. | https://arxiv.org/html/2404.15269v3 |
| 7 | Thread 2 ASTRA-bench 행: "주인공 수 표기 불일치(4 vs 5)" (미해결로 남김) | 불일치의 실체: **abstract는 "four protagonists"**, **본문은 "2,400 human-authored scenarios grounded in the digital lives of five unique protagonists"**(Dawei Shen, Theo Appleseed, John Quinn, Emily Rose, Lucas Garcia)로 5명이 이름까지 있습니다. 시나리오 수도 abstract 2,413 vs 본문 2,400으로 어긋납니다. 본문 기준(5명)을 쓰고 각주로 abstract 불일치를 적으면 됩니다. | https://arxiv.org/abs/2603.01357 ; https://arxiv.org/html/2603.01357 |
| 8 | Thread 5 CI-Bench 행: "44,100 test cases" + Caveats "CI-Bench 저자 목록 확인 필요" | 본문 "a total of **44,100** test cases, with a negative to positive label ratio of 7.4:1"로 맞습니다(abstract는 "44,000"). 저자: Zhao Cheng, Diane Wan, Matthew Abueg, Sahra Ghalebikesabi, Ren Yi, Eugene Bagdasarian, Borja Balle, Stefan Mellem, Shawn O'Banion. "무상태(stateless)" 표현도 본문에 명시되어 있습니다. Caveat 삭제 가능. | https://arxiv.org/html/2409.13903 ; https://arxiv.org/abs/2409.13903 |

---

## 2. Likely wrong / unsupported

| # | Report text (quoted) | Why doubtful | What to check |
|---|---|---|---|
| 1 | "**MobileFineTuner** — Geng et al. (arXiv 2512.08211, MobiSys 2026 채택으로 보고됨)" | MobiSys 2026 공식 accepted papers 페이지에 "MobileFineTuner"가 **없습니다**. 같은 페이지에 있는 온디바이스 fine-tuning 논문은 "FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs"입니다. arXiv 페이지에도 venue 표기가 없습니다(v1 제목 "A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones", v2 "A Mobile-Native Framework…"). | https://www.sigmobile.org/mobisys/2026/accepted_papers/ 에서 직접 재확인. 채택 근거가 없으면 "arXiv preprint"로 표기하고 H3 근거는 FBLayout으로 교체. |
| 2 | MobileFineTuner 행: "경험칙: FP16 학습은 1B 파라미터당 16GB 필요, 2025년 폰 RAM은 4–16GB" | arXiv abstract에는 이런 수치가 없습니다(본문 미확인). "1B당 16GB"는 FP16 full-FT + Adam 상태 기준 통상치(16 bytes/param)이지 이 논문 고유 기여가 아니므로 출처를 논문으로 달면 안 됩니다. | 본문에서 해당 문장을 찾거나, 일반 경험칙으로 표기. |
| 3 | Thread 4 Pielot 행: "메신저 알림을 6.15분(중앙값) 내에 확인하는지 … precision 81.2%" | ACM DL 403, Semantic Scholar abstract elided, 저자 PDF는 텍스트 추출 실패로 **확인 불가**. 제 기억으로는 이 논문의 주 지표가 "accuracy 70.6%"인데 확신할 수 없어 단정하지 않습니다. | https://dl.acm.org/doi/10.1145/2556288.2556973 또는 https://pielot.org/pubs/Pielot2014-CHI-AttPred.pdf 를 열어 라벨 정의(몇 분 내 확인)와 accuracy/precision을 다시 적을 것. |
| 4 | Attelia 행: "통제 실험, 30명×16일 in-the-wild … 민감 사용자에서 좌절 28%↓" | 위 §1-4와 같은 이유로 in-the-wild 규모와 28% 수치 확인 불가. Attelia II abstract에서 확인된 수치는 71.8%뿐. | UbiComp'15 논문 본문(§Evaluation) 재확인. |
| 5 | Thread 5 P³ 행 "(SIGIR 2026)" | arXiv 페이지에는 venue가 없고, 검색 스니펫(SIGIR '26 Melbourne)에서만 확인됨. 90.3–95.7% 수치는 abstract에서 확인. | https://sigir2026.org/en-AU/pages/program/accepted-papers 목록에서 제목 확인. |
| 6 | Thread 1 KnowU-Bench 라벨: "intervene / consent / silence, 거절 후 자제(post-rejection restraint)" | 본문의 실제 지표 이름은 **Act rate / Silent rate / Stop rate** 3개입니다. "consent"는 지표가 아니라 사용자 시뮬레이터의 clarification 대화에서 나오는 개념으로 보입니다. §(b) 2순위 근거로 "consent 라벨을 그대로 차용"한다고 썼으므로 이름을 맞춰야 합니다. | https://arxiv.org/html/2604.08455 §Metrics. |
| 7 | Thread 1 SentinelBench 행: "GitHub microsoft/sentinel_environments" | arXiv 페이지에서 코드 링크를 확인하지 못했습니다(3 models, 2 harness, 100 tasks, 10 apps는 확인). | GitHub 저장소 존재 여부 직접 확인. |
| 8 | Thread 2 iOSWorld 행 모델 열: "Qwen 3.5 35B 등"; 지표 열 "최고 51.9%, 다중 앱 36.7%" | 수치는 맞지만 **최고 51.9%/36.7%는 Claude Opus 4.6(vision+XML)**이고 Qwen3.5-35B-A3B는 10.5%입니다. 현재 표기는 Qwen이 51.9%인 것처럼 읽힙니다. | 모델 열에 "Claude Opus 4.6 (best), Qwen3.5-35B-A3B 10.5%"로 수정. |
| 9 | Thread 1 ProAgentBench 행: "VLM 판별 → 자원자 교정 → 규칙 필터의 3단계" 프라이버시 파이프라인, "burstiness B=0.787" | 0.787, 28,000+, 500+시간, "real > synthetic" 문구는 abstract에서 확인. 3단계 프라이버시 파이프라인은 본문 미확인. 코드는 "anonymous repository"로만 공개. | 본문 §Data collection 확인. |
| 10 | Thread 2 Gaia2 행: "Kimi-K2는 OpenReview 초록에서 21%, camera-ready 본문에서 20%" | 본문 표는 **20.1**입니다. 21 vs 20은 반올림 차이일 뿐 "출처마다 다름"으로 쓸 사안이 아닙니다. | 한 줄로 "20.1 (abstract 21%)"로 정리. |
| 11 | Key Finding 5 / Thread 4 AHs 2026 행: "interruptibility, 집중도, 짜증 (각 1–7)" | N=21, 사무실 환경, "providing the model with the history of users' previous ratings led to the strongest improvements"는 확인. 세 가지 평정 항목과 1–7 척도는 abstract에 없어 미확인. 저자(Lingler, Frijns, Boess, Murziakova, Wintersberger)를 표에 추가할 것. | ACM DL 본문. |
| 12 | Thread 5 SAPA-Bench 행 "(2025)" | 논문 제목은 "Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents"(arXiv 2508.19493)이며 GitHub에 **AAAI 2026** 채택 표기. 7,138 시나리오·3단계 민감도는 확인. | 제목·venue 보완. |
| 13 | 보조표 Kruger 행 (venue 없음) / FingerTip 행 (venue 없음) / ProactiveMobile "(CVPR 2026)" | Kruger et al.은 **CHIRA 2025** 채택, FingerTip 20K는 **ICLR 2026 poster**, ProactiveMobile은 GitHub(xiaomi-research/proactive-mobile)에 CVPR 2026 표기 — 확인됨. 표에 venue를 채우면 됩니다. | — |
| 14 | Thread 3 "PersonaMem, PersonaLens (Findings ACL 2025)" | PersonaMem은 GitHub에 **COLM 2025**로 표기. PersonaLens venue는 미확인. | 각각 분리 표기. |
| 15 | H2 근거: "AirGapAgent는 데이터 최소화기를 분리해 보호율 97%를 얻었습니다." | 수치(94%→45%, 97%)는 맞으나 AirGapAgent의 분리는 **프라이버시 최소화기 vs 대화 모델**이지 "filter(언제 개입) vs output(무엇을)" 분리가 아닙니다. H2의 직접 근거가 아니라 유추 근거임을 명시해야 합니다. | — |

---

## 3. Missing prior work

| # | Paper (title, authors, venue, year, URL) | Which thread | Why it matters for the gap claim |
|---|---|---|---|
| 1 | **ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation** — ICLR 2026, arXiv 2509.21730, https://arxiv.org/abs/2509.21730 | Thread 1 + 3 | proactive(타이밍) + personalized를 **한 시뮬레이션 안에서** 결합하고, 사용자 에이전트의 평점으로 assistant가 **시간에 따라 개선되는 곡선**을 보고합니다(32 personas). 보고서의 "proactive 연구는 개입 여부만 본다"와 "선호 학습 곡선을 재는 benchmark는 없다"는 서술을 모두 약화시킵니다. Δt=✓인 유일한 proactive 연구이므로 반드시 표에 넣어야 합니다. |
| 2 | **Preference-Driven Online Adaptation for Personalized Interaction Initiation in Proactive AI Assistants (EOPA)** — arXiv 2608.04416, https://arxiv.org/abs/2608.04416 | Thread 1/3/5, H1·H3 | "언제 개입할지"를 **온라인 피드백으로 재학습 없이** 갱신(evidence carriers + decision parameters), ProPerSim 기반 벤치마크에서 timing F1 +19.80, 하루 적응 시간 11.41 s→0.39 s. (d)+(e) 조합이 이미 존재함을 보여주며, H3(야간 재학습)의 필요성에 대한 **반대 증거**입니다. |
| 3 | **After Talking with 1,000 Personas: Learning Preference-Aligned Proactive Assistants From Large-Scale Persona Interactions** — arXiv 2602.04000, https://arxiv.org/abs/2602.04000 | Thread 1/3/5 | 1,000 persona 시뮬레이션으로 timing·autonomy·communication style 선호를 학습하고, **온디바이스 activation steering**으로 개별 사용자 피드백에 적응(34명 인간 실험). 온디바이스 + 선호 적응 + proactive timing을 함께 다룹니다. |
| 4 | **Proactive Service Agents: A Unified Decision Framework, Methods, and Evaluation** — arXiv 2609.03727, https://arxiv.org/abs/2609.03727 | Thread 1, (b) | 개입을 **silent / ask / assist / act** 4지선다로 두고 interruption cost, "option value of waiting"(= 지연의 가치), overreach·privacy cost를 명시한 POMDP 프레임워크. 보고서의 "urgency를 비용으로 정량화하거나 행동 옵션을 두는 연구는 없다"는 문장과 정면으로 겹치므로 positioning이 필요합니다(차이: 마감 함수, digest, 실행 검증은 없음). |
| 5 | **π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows** — arXiv 2605.14678, https://arxiv.org/abs/2605.14678 | Thread 1 | 100 multi-turn tasks × 5 personas, cross-session continuity, "task completion ≠ proactivity" 분리 측정. 이름부터 "proactive personal assistant"인데 빠져 있습니다. |
| 6 | **Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents (ATRBench)** — arXiv 2605.28108, https://arxiv.org/abs/2605.28108 | Thread 3 | "나중 세션에 필요한 선호를 지금 물어볼지"를 평가. 8개 frontier agent가 oracle 대비 ≥62점 낮음, 병목은 acquisition. 보고서의 "저녁 digest 질문 1회" 설계와 직결되며, H1(선호 추출)의 전제인 **선호 획득 능력이 약하다**는 반대 증거입니다. |
| 7 | **Batching smartphone notifications can improve well-being** — Fitz, Kushlev, Jagannathan, Lewis, Paliwal, Ariely, *Computers in Human Behavior* 2019 (CHI 아님), https://www.sciencedirect.com/science/article/abs/pii/S0747563219302596 | Thread 4 | n=237 무작위 현장실험: 하루 3회 batching은 주의·생산성·통제감↑, 매시간 batching은 효과 없음, 완전 차단은 불안·FoMO↑. "batch_to_digest" 행동과 digest 빈도 설계의 **유일한 실증 근거**인데 누락되었습니다. |
| 8 | **From Overwhelmed to Overview: Understanding Smartphone Users' Preferences and Expectations in Relieving Notification Overload via Text Summarization** — Chen et al., Proc. ACM HCI (MobileHCI) 2025, https://dl.acm.org/doi/10.1145/3743703 | Thread 4 | ChatGPT 기반 알림 요약 앱을 1주 in-the-wild 배포 + 20명 인터뷰. 사용자가 요약에서 우선시하길 원하는 알림 유형 3가지, 정보 공개 수준 3단계를 보고. "digest" 내용 설계의 HCI 근거. |
| 9 | **PersoNo: Personalised Notification Urgency Classifier in Mixed Reality** — Zheng et al., ISMAR 2025, arXiv 2508.19622, https://arxiv.org/abs/2508.19622 | Thread 4, C1 | LLM 기반 **개인화 urgency 분류기**(18명, self-labelled + interaction 데이터, 81.5% acc, FN 0.381). "urgency 라벨" 선행 정의로 인용해야 하며, activity context가 content·sender와 동등한 가중치라는 결과는 C1의 `user_context` 라벨을 뒷받침합니다. |
| 10 | **Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch** — Kim et al., arXiv 2602.03183, https://arxiv.org/abs/2602.03183 | 보조표(합성 개인 데이터) | 1.4M 합성 개인 기록(캘린더·문자·의료·법률·금융), 55.1M 속성 주석. 타임라인 일관성은 없지만 "합성 개인 데이터 생성기" 서베이에서 규모상 빠질 수 없습니다. |
| 11 | **ToolSandbox** — Lu et al. (Apple), arXiv 2408.04682, https://arxiv.org/abs/2408.04682 | Thread 2 | 상태 기반 개인 앱 도구 + 내장 LLM 사용자 시뮬레이터 + milestone 평가. τ-bench 옆에 있어야 할 Apple의 표준 환경. |
| 12 | **UserBench: An Interactive Gym Environment for User-Centric Agents** — Qian et al., arXiv 2507.22034 (venue 미확인), https://arxiv.org/abs/2507.22034 ; **UserToolBench** — arXiv 2608.10042, https://arxiv.org/abs/2608.10042 | Thread 3 | 숨겨진 선호를 상호작용으로 드러내야 하는 gym(모델이 선호의 <30%만 발견). UserToolBench는 hidden profile + tool-use 의사결정(10 profiles, 1,065 turns). KnowU-Bench와 같은 "profile 숨김" 계열. |
| 13 | **MyPCBench** — Jang et al., arXiv 2606.16748, https://arxiv.org/abs/2606.16748 ; **PSPA-Bench** — arXiv 2603.29318, https://arxiv.org/abs/2603.29318 | Thread 2 | iOSWorld 저자들의 데스크톱판(persona 1명, 184 tasks, Claude Opus 4.6 55.4%); PSPA-Bench는 12,855 개인화 지시·22 앱. "개인 데이터 위 반응형 실행" 열에 추가. |
| 14 | **HorizonBench** (보고서에 arXiv ID만 있음) — Li et al., arXiv 2604.17283, https://arxiv.org/abs/2604.17283 | Thread 3 | 360 simulated users × 6개월, 최고 모델 52.8%, 대부분 chance 이하. "변화하는 선호"의 어려움을 수치로 제공 — H1 리스크 근거로 승격할 것. |
| 15 | **FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs** — MobiSys 2026 (accepted list), https://www.sigmobile.org/mobisys/2026/accepted_papers/ ; **MobiLLM** (arXiv 2502.20421) / **PAE MobiLLM** (2507.01216) server-assisted side-tuning | Thread 5, H3 | peer-reviewed MobiSys 2026 온디바이스 fine-tuning 논문. MobileFineTuner 대신 H3 근거로 사용 가능. (본문은 열지 않았으므로 수치는 직접 확인 필요.) |
| 16 | **Sensing What Surveys Miss: Understanding and Personalizing Proactive LLM Support by User Modeling** — Liu, Karoui, Draxler, Kreuter, Chiossi, CHI 2026, https://arxiv.org/abs/2602.00880 | Thread 4 | 생리신호+마우스로 개인화 threshold를 맞춘 proactive LLM 지원; "정렬된 적응형 타이밍"이 품질·경험을 보존. LLM 시대 개인화 타이밍의 CHI 2026 근거. |
| 17 | **Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security** — Li et al. (Tsinghua AIR), arXiv 2401.05459, https://arxiv.org/abs/2401.05459 | Thread 5 | 온디바이스 개인 에이전트 서베이. related work 서두의 표준 인용. |
| 18 | 산업 기능: **Apple Intelligence Priority Notifications** (iOS 18.4, 온디바이스 중요도 판정) https://support.apple.com/guide/iphone/summarize-notifications-reduce-interruptions-iph1fbe7d2b9/ios ; **Samsung Galaxy AI Now Brief / Now Nudge** (일일 digest, Gemini Nano 온디바이스) ; **Google Pixel 10 Magic Cue / Daily Hub** (앱 간 맥락으로 선제 제안) | Thread 1/4/5 | urgency 순위화 + 일일 digest + 온디바이스가 **이미 상용화**되어 있습니다. 벤치마크가 없다는 주장은 유지되지만, "행동 옵션으로 digest를 두는 연구는 없다"는 문장은 상용 제품을 언급하며 "학술 benchmark로 평가된 적 없다"로 바꿔야 합니다. |
| 19 | (보조) **Toward User-Conditioned Evaluation of Personal LLM Agents under Temporal Interventions** — arXiv 2607.21635 (KDD 2026 workshop position), https://arxiv.org/abs/2607.21635 | Thread 2 | 같은 시간 개입을 서로 다른 사용자 상태에 replay해 오류 전파를 재자는 제안. 평가 설계 논의에 인용 가능(실증 없음). |

검색만 하고 열지 않은 후보(본인 확인 필요): "Parameter Efficiency Is Not Memory Efficiency" (arXiv 2604.22783), "Do Phone-Use Agents Respect Your Privacy?" (2604.00986), "PriMobiBench" (2609.13873), "Communication Policy Evolution for Proactive LLM Agents" (2606.14314), "ProactiveEval" (2508.20973), "HAS-Bench" (2607.04329).

---

## 4. Gap statement — does it survive?

엄격한 5중 결합(일관된 다중 앱 합성 타임라인 × 마감·지연비용 urgency × 4지 행동 × 검증 가능한 대리 실행 × 교정 학습 곡선 × 온디바이스)으로는 여전히 선행 연구가 없습니다. 그러나 보고서가 "조각만 다룬다"고 한 부분 중 세 곳이 이미 겹칩니다. (i) **(b)+(d)+(e)**: ProPerSim(ICLR 2026)·EOPA·"1,000 Personas"는 proactive 타이밍 판단을 사용자 피드백으로 시간에 따라 개선하는 곡선을 보고하고, 뒤의 둘은 온디바이스 적응까지 다룹니다. (ii) **(b)의 4지 행동·비용 프레임**: "Proactive Service Agents"가 silent/ask/assist/act와 interruption cost·option value of waiting을 이미 정식화했고, Apple Priority Notifications와 Samsung Now Brief가 urgency 순위화와 일일 digest를 상용화했습니다. (iii) **(a)**: Privasis가 대규모 합성 개인 기록을, ASTRA-bench가 시간 일관성 verifier를 제공합니다. 따라서 novelty 문장은 "마감 함수로 정의된 urgency와 digest batching을 **행동으로** 채점하고, 대리 실행을 **환경 상태로 검증**하며, 교정 기반 학습 곡선을 **같은 타임라인 위에서** 재는 폐루프 benchmark"로 좁혀 쓰고, 위 세 계열을 각각 "부분 중첩"으로 명시해야 리뷰어 공격을 피할 수 있습니다.

---

## 5. Hypotheses H1/H2/H3 — verdict on the verdicts

- **H1 (선호만 추출) — 판정 "실현 가능성 높음"에 조건부 동의.** 인용된 CIPHER는 요약·이메일 **문체** 선호를 200라운드 시뮬레이션으로 학습한 것이고, P³는 프로필 **비공개 하 개인화**(선호 추출과 무관)입니다. 즉 근거가 가설과 정확히 맞지 않습니다. 더 직접적인 근거는 EOPA(피드백으로 timing 선호를 재학습 없이 갱신, +19.8 F1)와 "1,000 Personas"(activation steering)입니다. 반대 증거: Ask-Now-Use-Later(선호 **획득** 자체가 병목, oracle 대비 −62점), HorizonBench(변화하는 선호에서 최고 52.8%), KnowU-Bench("preference acquisition and intervention calibration"이 병목). 결론: 추출한 선호를 저장하는 것보다 **무엇을 언제 물어볼지**가 진짜 난제이므로, H1 평가에 "선호 획득 정책"을 명시적으로 넣어야 합니다.
- **H2 (filter/output 분리) — "근거가 가장 강함"에 동의하되 근거를 교체.** 가장 강한 증거는 보고서가 짧게만 언급한 TGL 논문입니다: 14개 backbone 전부에서 F1 +16.7(최대 +46.0), LLM-as-trigger 대비 4–83× 빠름, **~220 MiB 온디바이스** — 보고서의 "수 MB~수백 MB 소형 filter" 주장을 정확히 뒷받침합니다. PRPF는 ProactiveMobile에서만 검증되었고, ProAgentBench의 계층 분해는 **평가 프레임**이지 분리가 더 낫다는 실험 증거가 아닙니다. AirGapAgent는 프라이버시 유추 근거입니다. 반대 증거: AHs 2026은 맥락 특징이 "사용자 평정 이력 이상의 정보를 거의 주지 않는다"고 보고하므로, 소형 filter의 입력은 센서 맥락보다 **개인 이력**이어야 하며, 선호를 모르는 filter는 KnowU-Bench가 말한 calibration 실패를 그대로 안습니다(filter가 선호 메모리를 읽어야 함 → H1과 결합 필수).
- **H3 (야간 재학습) — "조건부 실현 가능"에 동의하나 근거 수치가 틀려 있음.** FwdLLM의 LLaMA-7B INT4 peak은 4.0GB(1.5GB 아님)이고 연합학습 설정이며, MobileFineTuner의 MobiSys 채택은 확인되지 않습니다. LoRA peak-memory 논문의 1.02GB는 FP32 서버 baseline 대비 INT4 조건입니다. 남는 실질 근거는 PocketLLM(OPT-1.3B 6.5GB, 미분 없는 MeZO), LoRA peak-memory(3B, 12GB 폰 검증), FBLayout(MobiSys 2026). 반대 증거: EOPA는 가중치 재학습 없이 하루 0.39 s로 timing 선호를 적응시켰고, 하루 교정 수 건으로는 gradient 신호가 희소하다는 보고서 자체 지적과 합쳐지면, H3는 "주 단위 LoRA 증류" 정도로 축소하고 **ablation의 한 arm**으로만 두는 것이 안전합니다.

---

## 6. Benchmark labels & metrics — issues

- **라벨 중복**: `urgency ∈ {U0..U3}`는 `deadline_t`·`cost_of_delay(t)`의 이산화입니다. 둘 다 정답 라벨로 두면 annotator 간 불일치가 두 곳에서 나옵니다. `deadline_t`+비용 함수를 1차 라벨로, urgency 등급은 파생 변수로 정의하십시오.
- **단일 "정답 행동"의 타당성**: My Phone and Me·AHs 2026·PersoNo가 보여주듯 수용성은 사용자 상태와 개인 이력에 크게 의존합니다. 이벤트당 하나의 gold action 대신 **(persona, user_context)별 비용 행렬 또는 허용 행동 집합**을 정답으로 두지 않으면 PVR·false interruption rate가 개인차와 라벨 오류를 구분하지 못합니다. Krippendorff α가 낮게 나올 가능성이 큰 지점입니다.
- **행동 집합 누락**: "사용자에게 지금 물어보기(ask)"가 없습니다. Proactive Service Agents 프레임워크와 PARE(Observe 모드에서 제안)는 ask를 별도 옵션으로 둡니다. interrupt_now와 ask는 비용 구조(응답 필요 여부)가 다르므로 5번째 행동으로 추가하거나 배제 이유를 써야 합니다.
- **Net utility의 λ**: λ를 고정하면 순위가 λ에 종속됩니다. persona별 λ(민감 사용자)를 두거나 λ-sweep Pareto 곡선으로 보고하십시오.
- **ECE**: LLM 에이전트는 4지 행동에 대한 확률을 내지 않습니다. 확률을 어떻게 얻을지(sampling k회? verbalized confidence?) 명시하지 않으면 계산 불가한 지표입니다.
- **Digest와 마감의 연결**: `batch_to_digest`로 보낸 항목의 `deadline_t`가 digest 시각보다 이르면 "미충족"으로 채점된다는 규칙이 없습니다. 또한 digest 시각 자체(고정 저녁 vs Attelia식 breakpoint)가 실험 변수인지 상수인지 정하십시오. Fitz 2019(3회/일 효과, 1회/시 무효)를 digest 빈도 설계 근거로 인용하십시오.
- **False interruption rate 정의**: modality(voice/banner/haptic)별 방해 비용이 다르므로 "방해"의 단위를 정의해야 합니다. 하루 방해 횟수는 이벤트 수로 정규화하십시오.
- **Report fidelity / Digest round-trip**: LLM-judge를 쓸 것이면 인간 일치율 목표(선행: ProactiveBench 91.80%, MCP-Persona 91.5%)와 검증 표본 크기를 미리 적으십시오.
- **C3 학습 곡선의 시뮬레이터 의존성**: 교정을 주는 것이 LLM 사용자 시뮬레이터라면 곡선은 시뮬레이터의 함수입니다. PARE처럼 사용자 모델 민감도 ablation을 필수 보고 항목으로 넣고, hard/soft 규칙을 분리해 PVR을 보고하십시오. Forgetting·over-generalization은 매일 고정 probe set을 재평가하는 continual-learning 프로토콜로 정의해야 측정 가능합니다.
- **Interruptibility 용어**: Attelia의 breakpoint는 UI-event 및 physical-activity 전환 경계로 정의됩니다. "작업 전환점"이라고만 쓰면 부정확하며, "인지 부하 46%↓"는 실험실 결과임을 명시하십시오.
- **인간 검증 계획의 약점**: (i) 연구실 내 5–10명 × 2주 ESM은 IMWUT/CHI 리뷰어가 표본 편향(insider)과 규모로 반려할 수준입니다(선행: InterruptMe·My Phone and Me 20명, AHs 21명, Fitz 237명). (ii) 더 근본적으로 **ESM은 "지금 방해받아도 되는가"를 재지만 합성 이벤트의 urgency 라벨은 참가자가 그 이벤트를 받지 않으면 평가할 수 없습니다.** 상관을 보고하려면 합성 시나리오를 참가자에게 replay(Wizard-of-Oz 또는 앱 내 가짜 알림)하고 4지 행동을 직접 고르게 하는 "라벨 타당성 연구"로 재설계해야 합니다. (iii) 외부 참가자 ≥20명, IRB, 보상, 수집 항목을 명시하십시오.
- **프라이버시 지표**: PrivacyLens의 CI 위반율은 정보 흐름 규범 위반이고, "엉뚱한 수신자에게 전송"은 safety/irreversible action 오류입니다. 두 지표를 분리하십시오.

---

## 7. Top 5 fixes, ranked by impact

1. **H3 근거 수치 교정**: FwdLLM LLaMA-7B INT4 peak = 4.0GB(연합학습), 1.5GB는 README 문구로만 표기; MobileFineTuner를 "MobiSys 2026 채택"에서 arXiv preprint로 내리고 FBLayout(MobiSys 2026)·LoRA peak-memory(조건 명시)·PocketLLM으로 교체.
2. **Gap 문장·TL;DR 완화 + 누락 논문 추가**: ProPerSim(ICLR 2026), EOPA, "1,000 Personas", Proactive Service Agents 프레임워크, π-Bench, Ask-Now-Use-Later를 Thread 1/3에 넣고, "교정 학습 곡선을 재는 benchmark는 없다"→"proactive 실행·마감·digest와 결합해 잰 benchmark는 없다"로 수정. 상용 기능(Apple Priority Notifications, Samsung Now Brief, Pixel Magic Cue)을 한 문단으로 positioning.
3. **표의 사실 오류 수정**: PRPF 벤치마크(ProactiveMobile), PersonaBench 1,515 characters + Findings ACL 2025, KnowU 지표명(Act/Silent/Stop), iOSWorld 최고 모델(Claude Opus 4.6), Attelia 46%의 출처(PerCom 2015 실험실), ASTRA 주인공 5명(본문)·2,400 vs 2,413, CIPHER·CI-Bench caveat 삭제, venue 보완(FingerTip ICLR 2026, SAPA-Bench AAAI 2026, Kruger CHIRA 2025, P³ SIGIR 2026, PersonaMem COLM 2025).
4. **Thread 4 보강**: Fitz et al. 2019(batching 현장실험), Chen et al. 2025(LLM 알림 요약 in-the-wild), PersoNo(개인화 urgency 분류기), Liu et al. CHI 2026을 추가하고 Pielot 수치를 원문으로 재확인.
5. **벤치마크 설계 수정**: urgency/deadline 중복 제거, 정답을 (persona, context) 조건부 비용 행렬로 변경, "ask" 행동 추가 여부 결정, digest–deadline 채점 규칙 명시, ECE 산출법 명시, 인간 검증을 외부 ≥20명 replay 기반 라벨 타당성 연구로 재설계.

---

## 8. Sources opened

아래는 WebFetch로 실제 열어 내용을 확인한 URL입니다(검색 결과 스니펫만 본 항목은 제외).

arXiv abstract/HTML
- https://arxiv.org/abs/2602.11964 , https://arxiv.org/html/2602.11964 (Gaia2)
- https://arxiv.org/abs/2404.15269 , https://arxiv.org/html/2404.15269v3 (PRELUDE/CIPHER)
- https://arxiv.org/abs/2603.01357 , https://arxiv.org/html/2603.01357 (ASTRA-bench)
- https://arxiv.org/abs/2512.08211 (MobileFineTuner)
- https://arxiv.org/abs/2409.13903 , https://arxiv.org/html/2409.13903 (CI-Bench)
- https://arxiv.org/abs/2410.12361 (ProactiveBench)
- https://arxiv.org/abs/2505.14668 (ContextAgent)
- https://arxiv.org/abs/2602.04482 (ProAgentBench)
- https://arxiv.org/abs/2604.00842 , https://arxiv.org/html/2604.00842 (PARE)
- https://arxiv.org/abs/2604.08455 , https://arxiv.org/html/2604.08455 (KnowU-Bench)
- https://arxiv.org/abs/2606.05342 (SentinelBench)
- https://arxiv.org/abs/2602.21858 (ProactiveMobile)
- https://arxiv.org/abs/2606.09764 , https://arxiv.org/html/2606.09764 (iOSWorld)
- https://arxiv.org/abs/2606.02470 , https://arxiv.org/html/2606.02470 (MCP-Persona)
- https://arxiv.org/abs/2502.20616 , https://arxiv.org/html/2502.20616 (PersonaBench)
- https://arxiv.org/abs/2606.19528 , https://arxiv.org/html/2606.19528 (LoRA peak memory)
- https://arxiv.org/abs/2505.17615 , https://arxiv.org/html/2505.17615 (BehaviorGen)
- https://arxiv.org/abs/2606.03236 (PRPF)
- https://arxiv.org/abs/2605.30152 , https://arxiv.org/html/2605.30152 (TGL trigger)
- https://arxiv.org/abs/2507.21071 (FingerTip 20K)
- https://arxiv.org/abs/2308.13894 , https://arxiv.org/html/2308.13894 (FwdLLM; PDF도 저장했으나 로컬 텍스트 추출 실패)
- https://arxiv.org/abs/2409.00138 (PrivacyLens)
- https://arxiv.org/abs/2405.05175 (AirGapAgent)
- https://arxiv.org/abs/2502.09597 (PrefEval)
- https://arxiv.org/abs/2601.17569 (P³)
- https://arxiv.org/abs/2509.21730 (ProPerSim)
- https://arxiv.org/abs/2608.04416 (EOPA)
- https://arxiv.org/abs/2605.14678 (π-Bench)
- https://arxiv.org/abs/2605.28108 (Ask Now, Use Later)
- https://arxiv.org/abs/2606.16748 (MyPCBench)
- https://arxiv.org/abs/2608.10042 (UserToolBench)
- https://arxiv.org/abs/2508.19622 (PersoNo)
- https://arxiv.org/abs/2602.04000 (1,000 Personas)
- https://arxiv.org/abs/2602.03183 (Privasis)
- https://arxiv.org/abs/2603.29318 (PSPA-Bench)
- https://arxiv.org/abs/2604.17283 (HorizonBench)
- https://arxiv.org/abs/2607.21635 (User-conditioned evaluation, position)
- https://arxiv.org/abs/2609.03727 (Proactive Service Agents)

GitHub / 학회 / 기타
- https://github.com/UbiquitousLearning/FwdLLM
- https://github.com/deepakn97/pare
- https://www.usenix.org/conference/atc24/presentation/xu-mengwei (FwdLLM ATC'24 abstract)
- https://www.sigmobile.org/mobisys/2026/accepted_papers/ (MobileFineTuner 없음, FBLayout 있음)
- https://arize.com/blog/meta-ai-researcher-explains-are-and-gaia2/ (Mialon 인용 확인)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3795011.3795067 (AHs 2026, abstract 확인)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/2556288.2556973 (Pielot 2014, abstract elided)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/2750858.2807517 (Attelia II, abstract elided)

열기 실패(403/404/DNS): https://dl.acm.org/doi/10.1145/2556288.2556973 , https://dl.acm.org/doi/10.1145/2750858.2807517 , https://dl.acm.org/doi/10.1145/3795011.3795067 , https://dl.acm.org/doi/10.1145/3743703 , https://www.sigmobile.org/mobisys/2026/program.html , https://www.ht.sfc.keio.ac.jp/~slash/research/attelia/ , https://sigir2026.org/en-AU/pages/program/accepted-papers (목록 truncated), https://pielot.org/pubs/Pielot2014-CHI-AttPred.pdf 및 https://people.cs.nycu.edu.tw/~armuro/pubs/chen-et-al-2025-mobilehci.pdf (PDF 텍스트 추출 실패).

검색 스니펫으로만 확인(1차 출처 미개봉): ProactiveMobile CVPR 2026(GitHub 제목), ProactiveBench ICLR 2025 poster, Fitz 2019 서지, Chen 2025 MobileHCI 서지·N=20·1주, SAPA-Bench AAAI 2026, AgentSense AAAI 2026, IMUGPT 2.0 IMWUT DOI, Kruger CHIRA 2025, "Tell Me Why You're Asking" CHI 2026(33명), "Read the Room" CHI EA 2026, Liu et al. CHI 2026, P³ SIGIR 2026, Apple/Samsung/Google 기능 기사, TinyAgent EMNLP 2024 demo, PocketLLM PrivateNLP 2024 수치.
