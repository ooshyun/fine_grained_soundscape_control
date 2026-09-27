# 온디바이스 선제적(Proactive) 개인 비서 벤치마크 문헌·데이터셋 리뷰 (2023–2026)

**결론 먼저:** 다섯 가지를 한 벤치마크에서 함께 평가하는 선행 연구는 아직 없습니다. (1) 여러 앱에 걸친 일관된 합성 개인 타임라인, (2) 마감 시간이 있는 urgency 판단과 "지금 방해 / 조용히 처리 후 보고 / 저녁 digest로 묶기 / 무시" 네 가지 행동 선택, (3) 대리 실행(write-action)의 검증, (4) 사용자 교정을 통한 선호 학습과 그 learning curve, (5) 온디바이스 크기·프라이버시 제약입니다. 가장 가까운 기반은 Meta의 ARE/Gaia2(비동기 이벤트 기반 모바일 환경)와 그 위에 만든 PARE(능동 사용자 시뮬레이터)입니다. 여기에 ASTRA-bench식 "Pattern-of-Life" 생성 방법과 KnowU-Bench식 proactive 판단 체인 라벨을 얹는 구성을 권합니다.

## TL;DR
- **빈 조합:** 기존 연구는 조각만 다룹니다. ProactiveBench·ContextAgent·ProAgentBench는 "개입할지·언제"만, AppWorld·τ-bench·Gaia2·iOSWorld는 "실행"만, PRELUDE·PrefEval은 "선호"만, InterruptMe·Attelia는 "방해 가능성"만, PrivacyLens·AirGapAgent는 "유출"만 봅니다. 마감이 있는 urgency 판단, batching·digest, 교정을 통한 선호 학습, 온디바이스 제약을 한 번에 묶은 benchmark는 없습니다. 이것이 논문의 novelty 축입니다.
- **기반 추천(최대 3개):** ① ARE/Gaia2 Mobile + PARE를 환경·시뮬레이터로 쓰고, ② KnowU-Bench를 proactive·consent·restraint 라벨과 Android 실행 검증의 틀로 쓰고, ③ PRELUDE/CIPHER를 교정 기반 선호 학습 프로토콜로 씁니다. 합성 데이터 생성 방식은 ASTRA-bench의 PCFG 기반 pattern-of-life와 이벤트→다중 앱 artifact 투영 방식을 차용합니다.
- **지도교수 가설 평가:** (2) filtering 모델과 output 모델 분리는 근거가 가장 강합니다(PRPF, "Do Proactive Agents Really Need an LLM…", ProAgentBench의 계층 분해). (1) 원본 데이터 대신 선호만 추출하는 방식은 CIPHER·P³ 근거로 실현 가능하며, 1차 메커니즘으로 권합니다. (3) 야간 재학습은 기술적으로 가능해지고 있습니다(FwdLLM GitHub 기준 INT4/GPTQ 양자화 LLaMA-7B peak 1.5GB, LoRA peak memory 1.02GB까지 감소). 다만 대상은 소형 filter나 LoRA로 한정하고, 교정 데이터가 희소하므로 "선호 메모리 갱신 + 주기적 소형 모델 학습"을 병행하는 편이 현실적입니다.

---

## Key Findings

1. **"언제 개입할지" 연구는 2025–26년에 급증했지만 모두 제안(suggestion) 수준에 머뭅니다.** ProactiveBench(ICLR 2025)는 reward model로 수락·거절을 흉내 내고, ContextAgent(NeurIPS 2025)는 웨어러블 센서 맥락으로 필요성을 예측합니다. ProAgentBench(2026)는 실제 데이터로 timing을 예측하고, KnowU-Bench(2026)는 intervene·consent·silence를 봅니다. 그러나 urgency를 마감과 비용으로 정량화하거나 batching·digest를 행동 옵션으로 두는 연구는 없습니다.
2. **실행 환경은 성숙했고, 시간과 비동기성이 핵심 난제로 드러났습니다.** Gaia2에서 GPT-5(high)는 전체 42% pass@1로 최고였지만 time-sensitive tasks에서 실패합니다. Gaia2 공저자 Grégoire Mialon은 Arize 인터뷰에서 이렇게 설명했습니다: "A model like GPT-5 gets 0 on Time because it's extremely slow... missing a five-minute window by answering at six minutes." 생성이 즉시 끝난다고 가정하면 "GPT-5 jumps to ~34% on Time"입니다(inverse scaling). 이는 세탁기 기사 사례, 즉 정보는 있는데 제때 행동하지 못하는 문제가 최신 frontier 모델에서도 풀리지 않았다는 직접적인 증거입니다.
3. **합성 개인 데이터는 cross-app 일관성이 약점입니다.** Gaia2는 appendix에서 "temporal consistency across apps"를 미처리 한계로 명시합니다. ASTRA-bench만 verifier로 시간 일관성을 검사합니다. 여러 앱에 걸친 합성 타임라인의 realism을 실제 로그와 비교한 연구는 찾지 못했습니다.
4. **실제 데이터와 합성 데이터의 격차가 보고되고 있습니다.** ProAgentBench는 "real-world training data substantially outperforms synthetic alternatives"라고 보고합니다. 따라서 합성 벤치마크에 소규모 in-the-wild 보정(ESM)을 붙여야 리뷰어를 설득할 수 있습니다.
5. **UbiComp 전통의 interruptibility 라벨은 ESM 자기보고, 응답 시간, 인지 부하(NASA-TLX류)로 정의되었습니다.** LLM 시대 연구(AHs 2026)에서는 "사용자의 이전 평정 이력"을 넣는 것이 가장 큰 개선을 주었습니다. 개인화가 핵심이라는 뜻입니다.
6. **프라이버시 벤치마크는 "유출"만 측정합니다.** PrivacyLens에서 GPT-4는 privacy-enhancing 지시를 주어도 25.68%, Llama-3-70B는 38.69% 유출했습니다. 대리 행동 중에 누구에게 무엇을 보내는지(for whom) 판단하는 문제와 직결되지만, urgency나 선호와 결합한 평가는 없습니다.

---

## (a) Related-Work 비교표 (5개 thread)

표기: **Src** = 데이터 출처(R=real, S=synthetic, H=hybrid), **HV** = 인간 검증 여부, **Δt** = 시간에 따른 개선 측정 여부

### Thread 1 — 합성 이벤트·맥락 기반 Proactive Agent

| 논문 (venue, 연도) | Src / 생성 방식 | 앱·도메인 | 라벨 | 지표 / HV | 모델·온디바이스 | 프라이버시 | Δt | 공개·규모 | 본 과제 대비 gap |
|---|---|---|---|---|---|---|---|---|---|
| **Proactive Agent / ProactiveBench** — Lu et al. (ICLR 2025, arXiv 2410.12361) | H: 실제 키보드·마우스·클립보드·브라우저 활동을 수집하고, LLM gym으로 이벤트 합성 | coding, writing, daily life (데스크톱) | 제안 task 수락/거절 | F1, false alarm. 인간 annotation으로 학습한 reward model이 인간 판단과 91.80% F1 일치 | Qwen2-7B, LLaMA-3.1-8B fine-tune. 최고 F1 66.47% (GPT-4o 64.60%) | 없음 | ✗ | GitHub thunlp/ProactiveAgent. train 6,790 / test 233 이벤트 | 실행 없이 제안만. urgency·마감 없음. 모바일 앱 아님 |
| **ContextAgent / ContextAgentBench** — Yang et al. (NeurIPS 2025, arXiv 2505.14668) | H: 웨어러블 egocentric video·audio를 VLM으로 맥락 추출 + persona | 9개 일상 시나리오, 20개 tool | 선제 서비스 필요성, tool call | accuracy (baseline 대비 최대 +8.5% / +6.0%) | reasoning trace를 distill해 fine-tune | 명시적 처리 없음 | ✗ | 1,000 샘플 | 개인 앱 데이터(이메일·캘린더) 없음. batching 없음 |
| **ProAgentBench** — Tang et al. (arXiv 2602.04482, 2026) | R: 실제 사용자 화면 세션 500+시간, 1Hz 수집 | 업무용 PC 작업 | when to assist(timing), how to assist | timing 예측 + 내용 생성. real > synthetic 입증 | LLM/VLM baseline. memory·RAG 비교 | VLM 판별 → 자원자 교정 → 규칙 필터의 3단계 | ✗ | 28,000+ 이벤트, burstiness B=0.787 | 개인 통신 앱 없음. 행동 실행 없음 |
| **FingerTip 20K** — Yang et al. (arXiv 2507.21071, 2025) | R: 장기 Android 사용 로그 | 모바일 앱 전반 | 선제 task 제안 + 개인화 실행 | action matching + LLM judge | VLM 기반 | — | ✗ | 20K 에피소드급 | urgency·방해 비용 없음 |
| **ProactiveMobile** — Kong et al. (CVPR 2026, arXiv 2602.21858) | S + 전문가 30명 감사 | 14개 모바일 시나리오 | 맥락 기반 function sequence 예측 | success rate. fine-tuned Qwen2.5-VL-7B 19.15% vs o1 15.71% vs GPT-5 7.39% | 7B | — | ✗ | 3,660+ instances | 오프라인 예측. 실행·교정 없음 |
| **PARE / PARE-Bench** — Nathani et al. (UCSB·Apple·UW, arXiv 2604.00842, 2026) | S: ARE 위에 앱을 FSM으로 모델링한 능동 사용자 시뮬레이터. Stackelberg POMDP | communication, productivity, scheduling, lifestyle | Observe/Execute 모드. 제안 수락 여부 | 제안 품질, 효율, 실행 성공. 사용자 모델 민감도 ablation | frontier LLM | — | ✗ | 143 tasks, GitHub deepakn97/pare | urgency 라벨, 선호 누적 학습, 온디바이스 없음 |
| **KnowU-Bench** — Chen et al. (ZJU, arXiv 2604.08455, 2026) | S: Android emulator + profile에 기반한 LLM 사용자 시뮬레이터. profile은 숨기고 행동 로그만 노출 | 모바일 GUI 앱 | intervene / consent / silence, 거절 후 자제(post-rejection restraint) | rule 기반 + LLM judge. 초록 기준 명시적 실행에 강한 에이전트도 선호 추론·개입 보정이 필요한 모호한 지시에서는 "fall below 50%… even for frontier models like Claude Sonnet 4.6"이며, 병목은 "not GUI navigation but preference acquisition and intervention calibration". Claude Sonnet 4.6 실패의 80.0%가 개입·수동성 오류 | frontier + open VLM 11종 | — | ✗ | 42 일반 / 86 개인화 / 64 선제 tasks, 코드 공개 | 마감 기반 urgency, batching 없음. 교정 누적 없음 |
| **SentinelBench** — Maldaner et al. (Microsoft, arXiv 2606.05342, 2026) | S: 10개 합성 웹앱이 scripted event를 재생 | email, calendar, finance 등 | 조건 충족 시점 감지 | task completion, **reaction time**, resource use | browser agent | — | ✗ | 100 tasks, GitHub microsoft/sentinel_environments | "기다렸다가 제때 반응"을 측정하지만 사용자 방해 비용 개념 없음 |
| **PRPF (Perceive Before Reasoning)** (arXiv 2606.03236, 2026), **"Do Proactive Agents Really Need an LLM to Decide When to Wake…"** (arXiv 2605.30152, 2026) | 기존 benchmark(ProactiveAgent, FingerTip) 재사용 | — | 개입 여부(when)와 방법(how) 분리 | 오탐 억제, 효율 | 경량 trigger + LLM | — | ✗ | — | 가설 (2)의 직접 근거 |

### Thread 2 — 개인 앱 시뮬레이션 환경과 대리 실행

| 논문 | Src / 생성 | 앱 | 라벨 | 지표 / HV | 모델 | 프라이버시 | Δt | 공개·규모 | Gap |
|---|---|---|---|---|---|---|---|---|---|
| **Gaia2 + ARE** — Froger et al. (Meta, ICLR 2026, arXiv 2602.11964) | S: PersonaHub seed → 앱 간 dependency graph로 전파. universe당 400K–800K 토큰 | Messages, Chats, Emails, Calendar, Contacts, Shopping, Cabs, Files 등 12개 앱, 101 tools | write-action oracle. Execution / Search / Ambiguity / Adaptability / Time / Noise / A2A | write-action verifier(RLVR에 사용 가능). ICLR 2026 camera-ready 기준 인간 annotation과 "0.99 precision, 0.95 recall"로 일치. 시나리오는 인간이 annotation. GPT-5(high) 42% pass@1. Kimi-K2는 OpenReview 초록에서 21%, camera-ready 본문에서 20%로 출처마다 다름 | frontier 및 open | — | ✗ | 1,120 시나리오(unique 800), 10 universes, 오픈소스 | notification policy로 선제 행동 연구는 가능하지만 urgency·선호 라벨 없음. 앱 간 시간 일관성은 미처리(appendix에서 명시) |
| **AppWorld** — Trivedi et al. (ACL 2024, arXiv 2407.18901) | S: 가상 인물의 디지털 활동으로 앱을 채움 | 9개 일상 앱(메일, 결제, 메시지 등), 457 APIs | state 기반 unit test | task goal completion | code agent | — | ✗ | 750 tasks | 반응형(reactive) 지시만 있음 |
| **τ-bench / τ²-bench** — Yao et al. 2024 (arXiv 2406.12045); Barres et al. 2025 | S: LLM 사용자 시뮬레이터 + 정책 문서 | retail, airline, telecom | DB 최종 상태 | pass^k(신뢰성) | frontier | — | ✗ | 공개 | 고객센터 도메인. 개인 데이터 아님 |
| **AndroidWorld** — Rawles et al. (ICLR 2025) | 실제 Android 앱 + 파라미터화된 task | 20개 앱 | 프로그램 기반 reward | success rate | VLM agent | — | ✗ | 116 task 템플릿 | 개인 이력 없음 |
| **iOSWorld** — Jang et al. (arXiv 2606.09764, 2026) | S: 단일 persona "Jordan Avery"를 26개 자체 제작 iOS 앱에 연결 | 거래, 메시지, 여행, 금융, 소셜 | 단일 앱 27 / 다중 앱 60 / 메모리·개인화 46 | rubric + judge. 최고 51.9%, 다중 앱 36.7% | Qwen 3.5 35B 등 | 전부 합성 | ✗ | 133 tasks, 오픈소스 | 반응형. persona 1명 |
| **ASTRA-bench** — Xiu et al. (Apple, arXiv 2603.01357, 2026) | S: 전기 + 소셜 multigraph + **PCFG 기반 pattern-of-life** → 이벤트를 다중 앱 artifact로 투영. Draft→Critique→Revise→Verify(시간 일관성 검사) | Contact, Calendar, Email, Message, WhatsApp, Phone call | tool trace, 상태 스냅샷 | 인간이 쿼리·성공 조건 작성. Claude-4.5-Opus macro 0.9112 | frontier | — | ✗ | 2,413 시나리오, 이력 평균 약 14일. 코드 "coming soon", CC BY-NC-ND. 주인공 수 표기 불일치(4 vs 5) | 반응형이지만 **합성 타임라인 생성 방식은 가장 참고할 만함** |
| **MCP-Persona** — Wang et al. (ICML 2026, arXiv 2606.02470) | H: 실제 MCP 서버 trace → LLM이 simulator 코드 생성, Context-Tree | Lark, Slack, Notion, email, SNS 등 24개 서버 | checkpoints | LLM judge와 인간 판단 91.5% 일치 | frontier | 민감 필드 가짜로 대체 | ✗ | 173 tasks / 970 checkpoints | 선제성, urgency 없음 |
| **PersonaBench** — Tan et al. (Salesforce, arXiv 2502.20616, 2025) | S: PersonaHub + social graph → 대화·AI 채팅·구매 이력. noise 0–0.7 | 메시지, 구매 | 개인 속성 QA | recall. GPT-4o는 정답 맥락을 주어도 0.444 | — | 합성 비공개 데이터 | 선호 갱신 확률 <1% | 582 문항, 15 characters | 이해만 평가하고 행동 없음. realism 검증 없음 |

### Thread 3 — 장기 메모리·개인화·교정 기반 선호 학습

| 논문 | Src | 도메인 | 라벨 | 지표 / HV | 모델 | Δt | 공개 | Gap |
|---|---|---|---|---|---|---|---|---|
| **PRELUDE / CIPHER** — Gao et al. (NeurIPS 2024, arXiv 2404.15269) | S: GPT-4 시뮬레이션 사용자가 잠재 선호에 따라 편집 | 요약, 이메일 작성 | 잠재 선호(정답) | 누적 edit distance(사용자 교정 비용), LLM query 비용. 학습된 선호와 정답의 유사도 | 프롬프트 기반, 가중치 학습 없음 | **✓ (누적 교정 비용 곡선)** | GitHub gao-g/prelude | 텍스트 스타일 선호뿐. 행동·타이밍 선호 없음 |
| **PrefEval** — Zhao et al. (ICLR 2025 Oral) | S + 수작업 큐레이션 | 20개 주제 | 명시적·암묵적 선호 위반 | zero-shot에서 10턴(약 3k 토큰)만에 정확도 <10%이지만 이는 "across most evaluated models" 기준. 본문 수치로는 5턴이면 정확도가 약 80%에서 30% 미만으로 떨어지고, 10턴에서 "GPT-o1 has 50% accuracy while Claude 3.5 Sonnet and Gemini 1.5 Pro have near-zero". fine-tuning으로 개선 | 10종 LLM | 부분(턴 수 함수) | 3,000 pairs, 공개 | 대화형만. 앱 행동 없음 |
| **LoCoMo** — Maharana et al. (ACL 2024) / **LongMemEval** — Wu et al. (ICLR 2025) | S: LLM으로 생성한 장기 다중 세션 대화 | 채팅 | QA, 시간 추론, knowledge update | 정확도 | — | 간접 | 공개 | 개인 앱 이벤트 아님 |
| **LaMP** — Salemi et al. (2023) / **PersonalLLM** — Zollo et al. (arXiv 2409.20296) / **PersonaMem**, **PersonaLens** (Findings ACL 2025), **HorizonBench** (arXiv 2604.17283) | R/S 혼합 | 글쓰기, 추천, 대화 | 사용자별 출력·선호 | 자동 지표 | 다양 | HorizonBench는 변화하는 선호를 다룸 | 공개 | 교정을 통한 행동 규칙 학습("엄마에게 아침엔 X")은 없음 |

### Thread 4 — Interruptibility·알림·주의(UbiComp/IMWUT/CHI)

| 논문 | 수집 방식(in-the-wild) | 라벨 정의 | 규모 | 결과 | 본 과제 시사점 |
|---|---|---|---|---|---|
| **InterruptMe** — Pejovic & Musolesi (UbiComp 2014, pp. 897–908) | 스마트폰 센서 + ESM | 알림에 반응했는지, 반응 시간, 감정, 참여도 | 실제 스마트폰 trace | activity·location·시간·감정·engagement가 서로 다른 측면의 interruptibility를 결정 | 라벨을 "반응성"과 "적절성"으로 분리해야 함 |
| **Designing Content-driven Intelligent Notification Mechanisms** — Mehrotra et al. (UbiComp 2015, pp. 813–824) | 로깅 + ESM | 알림 내용·발신자 기반 수용성 | — | 내용과 발신자가 수용성을 좌우 | "for whom"과 발신자 중요도가 urgency 입력으로 필요 |
| **My Phone and Me** — Mehrotra et al. (CHI 2016) | 자동 로깅 + ESM | 반응 시간, 체감 방해도 | 20명, 알림 10,372건, 설문 474건 | 표시 방식, 알림 유형, 발신자-수신자 관계, 과업 상태가 영향. 중요한 내용이라도 방해가 됨 | urgency와 방해 비용을 별도 축으로 라벨링 |
| **Didn't You See My Message?** — Pielot et al. (CHI 2014) | 폰 맥락 로깅 | 메신저 알림을 6.15분(중앙값) 내에 확인하는지 | — | precision 81.2% | 응답 지연 예측과 마감 결합 가능 |
| **Attelia / Attelia II** — Okoshi et al. (PerCom 2015; UbiComp 2015) 및 Pervasive and Mobile Computing | breakpoint(작업 전환점) 감지 | 인지 부하, 좌절감 | 통제 실험, 30명×16일 in-the-wild | 무작위 타이밍 대비 인지 부하 46%↓. 민감 사용자에서 좌절 28%↓ | "digest 시점"을 breakpoint로 선택하는 baseline |
| **Attention and Engagement-Awareness in the Wild** — Okoshi et al. (PerCom 2017) | Yahoo! JAPAN 앱 대규모 배포 | 반응 시간, 참여도 | 대규모 | breakpoint 전달이 효과적 | 대규모 검증 방법론 |
| **Interruptibility Prediction for Ubiquitous Systems** — Turner et al. (UbiComp 2015, 서베이) | — | 관행 정리 | — | 일반화된 접근은 거의 진전이 없음 | 라벨 표준화가 기여점이 될 수 있음 |
| **Using LLMs to Model Notification Timing from Context-Sensitive Features** (Augmented Humans 2026) | 사무실 작업 중 LLM이 알림 스케줄링 | interruptibility, 집중도, 짜증 (각 1–7) | N=21 | 기대만큼의 성능은 아니었음. **사용자 평정 이력 제공이 가장 큰 개선** | 개인화된 urgency·방해 모델의 필요성 |
| **"Tell Me Why You're Asking"** (CHI 2026), **Read the Room** (CHI EA 2026), **AttenTrack** (arXiv 2509.01414) | LLM 시대의 알림·그룹채팅 개입 | 선호 피드백 참여 | — | — | end-of-day 질문 설계에 참고 |

### Thread 5 — 프라이버시와 온디바이스

| 논문 | 내용 | 핵심 수치 | 시사점 |
|---|---|---|---|
| **AirGapAgent** — Bagdasarian et al. (CCS 2024) | contextual integrity에 기반해 task에 필요한 데이터만 에이전트에 노출 | context hijacking으로 Gemini Ultra 보호율 94%→45%. AirGapAgent는 97% | "filter/minimizer와 대화 모델 분리" = 가설 (2)의 프라이버시 측 근거 |
| **PrivacyLens** — Shao et al. (NeurIPS 2024 D&B) | seed(5-tuple) → vignette → agent trajectory | seed 493개. GPT-4 25.68%, Llama-3-70B 38.69% 유출 | 대리 메시지 전송 시 "for whom" 유출 지표로 차용 |
| **CI-Bench** (Google DeepMind, arXiv 2409.13903) | 합성 대화·이메일에 대한 CI 판단 | 44,100 test cases | 무상태(stateless)라 타임라인 없음 |
| **SAPA-Bench / PrivacyBench** 계열 (2025) | 스마트폰 에이전트의 프라이버시 판단 | 7,138 시나리오, 민감도 3단계 | 온디바이스 에이전트 평가 보조 |
| **TinyAgent** — Erdogan et al. (EMNLP 2024 Demo) | 1.1B/7B 소형 모델의 function calling, Tool RAG, 4-bit 양자화 | fine-tune한 SLM이 GPT-4-Turbo의 function calling 성능을 상회 | 온디바이스 실행기의 실현 가능성 |
| **FwdLLM** — Xu et al. (USENIX ATC 2024) | forward-only(perturbed inference) 연합 fine-tuning | 메모리 최대 93%↓, 라운드당 169MB, 에너지 96.7%↓는 모두 Pixel 7 Pro의 RoBERTa-large·YELP-P 측정값. LLaMA-7B는 INT4/GPTQ 양자화 조건에서 peak 1.5GB(GitHub UbiquitousLearning/FwdLLM)이며, 논문은 "fine-tuning... LLaMA over COTS smartphones within only 10 minutes"라고 보고 | 가설 (3)의 근거 |
| **PocketLLM** — Peng et al. (PrivateNLP 2024) | MeZO 방식 | OPPO Reno 6에서 RoBERTa-large 약 4GB, OPT-1.3B 약 6.5GB | 1B급 on-device 학습 가능 |
| **MobileFineTuner** — Geng et al. (arXiv 2512.08211, MobiSys 2026 채택으로 보고됨) | C++ 네이티브 Full-FT/LoRA, ZeRO식 sharding, 배터리 인지 스케줄러 | 경험칙: FP16 학습은 1B 파라미터당 16GB 필요, 2025년 폰 RAM은 4–16GB | 야간 충전 중 학습 스케줄링 |
| **LoRA Peak Memory Reduction** (arXiv 2606.19528) | 엣지 LoRA 메모리 기법 | Llama-3.2 3B, 2048 토큰: 26.20GB → 1.02GB | 3B 야간 LoRA가 수치상 가능 |
| **P³** — Salemi & Zamani (SIGIR 2026) | 온디바이스 소형 모델이 프로필 RAG로 서버 모델의 초안을 검증·수정. 프로필은 전송하지 않음 | 전체 프로필 유출 upper bound의 90.3–95.7% 회복 | 가설 (1)·(2)의 근거 |
| **POPI** (arXiv 2510.17881), **Profile-to-PEFT** (arXiv 2510.16282) | 자연어 선호 요약 / 프로필 → LoRA hypernetwork | — | 요약 자체도 유출될 수 있음을 경고 |

### (보조) 합성 개인 데이터 생성 — UbiComp 계열 포함

| 연구 | 생성 방식 | 실제 데이터 대비 검증 |
|---|---|---|
| **BehaviorGen** — Li et al. (Tsinghua, arXiv 2505.17615) | gpt-4o에 사용자 프로필 + 소수의 실제 이벤트를 주고 [요일, 시각, 장소, intent] 생성 | 합성 데이터만으로 fine-tuning 이득의 62.0% / 87.8% 회복. MIA 공격 성공률 <0.55 |
| **Synthetic Data Generation for Screen Time and App Usage** — Kruger et al. (arXiv 2509.13892) | ChatGPT 프롬프트 2×2 설계(persona 상세도 × seed 예시 유무) | 상세 프롬프트일 때 일부 용도에 feasible. fidelity와 diversity의 trade-off |
| **AgentSense** — Thukral et al. (AAAI, arXiv 2506.11773) | LLM persona·일과 → VirtualHome 실행 → 가상 센서 | 합성 + 소량 실제 데이터 ≈ 실제 데이터 전체 |
| **IMUGPT 2.0** — Leng et al. (IMWUT 2024) | LLM 텍스트 → motion → 가상 IMU | HAR 성능 향상 |
| **내부 문서** "Synthesized Dataset for proactive agent" (Google Drive, 2025-12 작성) | LlamaPIE(Proactive In-Ear Conversation Assistants) 방식: 2단계 system/user 프롬프트, 키워드×전문가 목록, 세션당 9개 개입 원칙 중 2개 샘플링 | 미검증. earable 대화용이며 다중 앱 타임라인은 아님 |

---

## (b) 재사용 기반 Shortlist (최대 3개)

1. **ARE/Gaia2 Mobile + PARE (1순위, 환경 기반)**
   - 이벤트 기반 비동기 환경이라 시뮬레이션 시간이 흐르는 동안 기사 메시지 같은 외부 이벤트가 도착합니다. notification policy로 에이전트의 관찰 범위를 조절할 수 있습니다.
   - 12개 앱과 101 tools가 이미 있습니다.
   - write-action verifier가 대리 실행을 검증하고 RLVR 보상으로도 쓸 수 있습니다.
   - PARE가 이 위에 FSM 기반 능동 사용자 시뮬레이터를 이미 얹었으므로 "사용자에게 물어보기, end-of-day digest 응답"을 구현할 수 있습니다.
   - 확장이 필요한 부분: urgency와 마감 라벨, 컨텍스트 상태(회의·수면·운전) 채널, 선호 규칙 저장소, 앱 간 시간 일관성 verifier(ASTRA-bench 방식).
2. **KnowU-Bench (2순위, 라벨 체계와 Android 실행 검증)**
   - profile을 숨기고 행동 로그만 노출하는 설계, intervene·consent·silence·post-rejection restraint라는 판단 체인을 그대로 차용할 수 있습니다.
   - 재현 가능한 Android emulator가 있어서 온디바이스 GUI 경로 실험도 가능합니다.
3. **PRELUDE/CIPHER 환경 (3순위, 교정 학습 프로토콜)**
   - 시뮬레이션 사용자가 잠재 선호에 따라 교정을 주고, 누적 교정 비용이 줄어드는 곡선을 측정합니다. 이는 본 과제의 "교정을 통한 개선" 축과 정확히 맞습니다.
   - 텍스트 edit을 행동 교정("8시 회의 잡지 마", "엄마에게 아침엔 보내지 마")으로 치환하면 됩니다.

*대안 참고:* ASTRA-bench는 생성 방법론이 가장 좋지만 코드가 아직 "coming soon"이고 CC BY-NC-ND입니다. 그래서 방법만 차용합니다. ProAgentBench는 실제 timing 데이터를 소규모 real 검증 세트로 보조 활용할 수 있습니다.

---

## (c) 벤치마크 라벨·지표 초안

**데이터 단위:** persona × 14–30일 타임라인. PCFG 기반 pattern-of-life로 만들고, 캘린더·메일·메시지·통화·사진 메타데이터로 투영합니다. 각 incoming event에 아래 라벨을 붙입니다.

### C1. Urgency 판단
- **라벨**
  - `urgency ∈ {U3 즉시, U2 수 시간 내, U1 당일, U0 무시 가능}`
  - `deadline_t`: 행동이 가치를 잃는 시각. 예: 기사 도착 +15분
  - `cost_of_delay(t)`: 단계 함수 또는 선형 함수
  - `sender_importance`
  - `user_context ∈ {중요 회의, 수면, 운전, 여가…}`
- **정답 행동**: `{interrupt_now(modality: voice/banner/haptic), act_silently_and_report, batch_to_digest, ignore}`. 운전 중이면 voice로 전달합니다.
- **지표**
  - **Deadline-met recall**(U3·U2 중 마감 전 처리 비율)
  - **Time-to-action / slack**
  - **False interruption rate** 및 **하루 방해 횟수**
  - 방해 비용 가중 **Net utility** = Σ(적시 처리 가치) − λ·Σ(부적절 방해 비용)
  - 기대 비용 기반 **calibration(ECE)**
- **인간 검증:** 3인 이상 annotator의 Krippendorff α. 소규모 in-the-wild ESM(InterruptMe나 My Phone and Me 방식)으로 체감 방해도와 합성 라벨의 상관을 보고합니다.

### C2. 대리 행동(Delegated Action)
- **라벨:** write-action oracle(ARE verifier 형식). 수신자(for whom), 인자(시간·장소), 되돌릴 수 있는지 여부(`reversible`).
- **지표**
  - **Action pass@1 / pass^k**
  - **Argument accuracy**(ASTRA-bench에서 payload가 병목으로 드러남)
  - **Harmful/irreversible action rate**
  - **Digest round-trip success**: 저녁 질문 1회 → 답변 → 자동 전송 완료율, 질문 압축률(요청 N건 / 질문 1회)
  - **Report fidelity**: 사후 보고가 실제 수행 로그와 일치하는지
  - **Privacy leakage rate**: PrivacyLens 방식 CI 위반 비율

### C3. 선호 학습·교정
- **라벨**
  - 각 persona에 숨겨진 선호 규칙 집합. hard(예: 8시 회의 금지)와 soft(오후 선호)로 구분하고, 범위(`scope`: 사람·시간대·앱)와 예외(긴급 시 무시 가능 여부)를 둡니다.
  - 교정 이벤트 스크립트
- **지표**
  - **Preference violation rate**(PVR)
  - **Corrections-to-convergence**: 동일 규칙 위반이 0이 될 때까지 필요한 교정 수
  - **Repeat-mistake rate**
  - **Over-generalization rate**: 예를 들어 "엄마에게 아침엔 X"를 배운 뒤 긴급 상황까지 막는 경우
  - **Forgetting**: 새 규칙 학습 후 기존 규칙 위반 증가
  - **누적 교정 비용 곡선**(PRELUDE 방식, x축 = 일수)
- **시스템 지표:** on-device 지연, 메모리, 에너지(mJ/event), 야간 업데이트 시간

---

## (d) Gap Statement

> 기존 연구는 선제적 개입 여부·타이밍 예측(ProactiveBench, ContextAgent, ProAgentBench, KnowU-Bench), 개인 앱 환경에서의 대리 실행(AppWorld, Gaia2, iOSWorld, ASTRA-bench), 대화형 선호 추종과 교정 학습(PrefEval, PRELUDE), 인간 interruptibility 모델링(InterruptMe, Attelia), 프라이버시 유출 평가(PrivacyLens, AirGapAgent)를 **각각 따로** 다뤘다. 다음을 **하나의 폐루프(closed-loop)로** 평가하는 benchmark는 없다: **일관된 다중 앱 합성 개인 타임라인** 위에서 **마감·지연 비용으로 정의된 urgency**에 따라 **즉시 방해 / 조용히 처리 후 보고 / digest로 묶기 / 무시**를 선택하고, 사용자 맥락(회의·수면·운전)에 맞는 modality로 **검증 가능한 대리 행동**을 수행하며, **사용자 교정으로부터 행동 선호를 누적 학습하는 곡선**을 **온디바이스 크기·프라이버시 제약 하에서** 측정하는 것.

---

## (e) 온디바이스 자기개선에 대한 지도교수 3가설 평가

**H1. 원본 데이터 대신 선호만 추출** — *실현 가능성 높음, 1차 메커니즘으로 권장*
- **근거**
  - CIPHER는 교정에서 자연어 선호 기술을 추론해 retrieve합니다. arXiv 2404.15269 v3 원문(Gao, Taymanov, Salinas, Mineiro, Misra)에 따르면 "CIPHER outperforms several baselines by achieving the lowest edit distance cost while only having a small overhead in LLM query cost"이며, 원본 edit을 넣는 것보다 프롬프트가 짧고 해석 가능하며 사용자가 수정할 수 있습니다.
  - P³는 프로필을 로컬에 두고도 전체 공유 upper bound의 90.3–95.7%를 회복했습니다.
- **리스크**
  - PrefEval에서는 프롬프트만으로는 긴 맥락에서 선호 추종이 무너지고 fine-tuning이 개선을 주었습니다.
  - POPI는 선호 요약 자체가 유출 대상이 될 수 있다고 경고합니다.
- **권고:** 선호는 온디바이스 구조화 규칙 메모리(scope·예외 포함)로 저장하고, 외부로 보내지 않습니다.

**H2. filtering 모델과 output 모델 분리** — *근거가 가장 강함*
- **근거**
  - PRPF와 "Do Proactive Agents Really Need an LLM to Decide When to Wake…"는 경량 trigger/filter로 비개입 사례를 먼저 걸러 오탐과 비용을 줄입니다.
  - ProAgentBench는 timing과 content를 계층적으로 분해합니다.
  - AirGapAgent는 데이터 최소화기를 분리해 보호율 97%를 얻었습니다.
  - UbiComp의 interruptibility 모델(InterruptMe, Attelia)은 원래 소형 분류기였습니다.
- **함의:** 개인화 학습은 **소형 urgency/filter 모델**(수 MB~수백 MB)만 온디바이스에서 자주 갱신하고, 실행용 LLM(1–3B)은 고정하거나 드물게 LoRA로 갱신합니다. 이렇게 하면 학습 자원 문제와 프라이버시 문제를 동시에 줄일 수 있습니다.

**H3. 야간 재학습(nightly retraining)** — *조건부로 실현 가능*
- **근거**
  - FwdLLM은 INT4/GPTQ 양자화 조건에서 LLaMA-7B 연합 fine-tuning을 peak 1.5GB로 보고했습니다(GitHub). 에너지 96.7% 절감은 Pixel 7 Pro의 RoBERTa-large 측정값입니다.
  - PocketLLM은 폰에서 OPT-1.3B를 약 6.5GB로 학습했습니다.
  - LoRA 메모리 기법으로 Llama-3.2 3B가 26.20GB에서 1.02GB까지 줄었습니다.
  - MobileFineTuner는 배터리 인지 스케줄러를 제공합니다.
- **한계**
  - 하루 교정 수는 수 건에 불과해 가중치 학습 신호가 희소하고 과적합이나 망각 위험이 큽니다.
  - 연합 방식은 원본을 보내지 않아도 gradient 유출 문제가 남습니다.
- **권고 설계:** "실시간 선호 메모리 갱신(H1) + 야간 소형 filter 학습(H2) + 주 단위 LoRA 증류"의 3계층 구성. 실험에서는 세 계층의 누적 교정 비용 곡선을 ablation으로 비교합니다.

---

## Recommendations (다음 4–6주)
1. ARE를 fork해 urgency·deadline·context 채널과 선호 규칙 저장소를 추가합니다. 기사 메시지 사례를 seed 시나리오 20개로 만듭니다.
2. ASTRA-bench식 PCFG pattern-of-life 생성기에 cross-app 시간 일관성 verifier를 붙여 persona 50명 × 14일을 생성합니다. 내부 LlamaPIE 방식 문서는 대화 생성 부분에만 재활용합니다.
3. 라벨 신뢰도 확보를 위해 연구실 내 5–10명이 2주 ESM 파일럿을 합니다. 이 데이터는 공개하지 않고 합성 라벨의 보정에만 씁니다. 이것이 "synthetic-only" 비판(ProAgentBench)에 대한 방어가 됩니다.
4. Baseline: frontier LLM(상한), 1–3B 온디바이스 LLM, 분리형(소형 filter + LLM), 규칙 기반(Attelia breakpoint + 우선순위 규칙).

## Caveats
- 2026년 arXiv 논문 다수(ProAgentBench, PARE, KnowU-Bench, iOSWorld, SentinelBench, ASTRA-bench, PRPF)는 확인 시점에 peer review 여부가 불분명합니다. 수치는 preprint 기준입니다.
- 다음 항목은 서지 확인이 추가로 필요합니다: MobileFineTuner의 MobiSys 2026 채택(2차 출처), CI-Bench 저자 목록, ASTRA-bench의 주인공 수 불일치, ContextAgent의 온디바이스 크기 세부.
- CIPHER의 교정 비용 감소율(요약 31%, 이메일 73%)은 2차 요약 출처에서만 확인했습니다.
- IMWUT·UbiComp에서 LLM으로 합성 ESM이나 알림 로그를 생성한 연구는 이번 조사에서 찾지 못했습니다. 이 부분 자체가 기여점이 될 수 있습니다.
