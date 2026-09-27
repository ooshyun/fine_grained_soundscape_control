# Survey cross-review kit (Opus × Fable)

`report.md`를 Opus와 Fable 서브에이전트가 각각 적대적으로 검토하고, 두 리뷰를 합쳐 보완 목록을 만드는 구성.

## 구조

```
review/
├── README.md                          ← 이 파일
├── report.md                          ← 검토 대상 (9/23 리서치 리포트)
├── REVIEWER_SYSTEM_PROMPT.md          ← 두 리뷰어가 공유하는 시스템 프롬프트(참고용 사본)
└── .claude/agents/
    ├── survey-reviewer-opus.md        ← model: opus
    └── survey-reviewer-fable.md       ← model: claude-fable-5-1
```

subagent 정의는 `.claude/agents/`에 두면 이 폴더에서 Claude Code를 열 때 자동 로드된다 (필요하면 `/agents`로 확인). `model` 필드는 `opus`/`sonnet`/`haiku` 별칭 또는 전체 모델 ID를 받으므로 Fable은 `claude-fable-5-1`로 지정했다. 계정에서 Fable 사용이 안 되면 그 subagent만 실패하니, 그 경우 `model: opus`로 바꿔 두 번 돌리거나 다른 접근 가능한 모델 ID로 바꾸면 된다.

## 실행

```bash
cd review
claude
```

Claude Code 안에서 아래를 그대로 붙여넣기:

```
report.md를 두 서브에이전트로 병렬 검토해줘.
- survey-reviewer-opus → 결과를 reviews/review-opus.md 에 저장
- survey-reviewer-fable → 결과를 reviews/review-fable.md 에 저장
각 서브에이전트는 REVIEWER_SYSTEM_PROMPT의 체크리스트 6개를 전부 수행하고, 출력 형식을 지켜야 해.
둘 다 끝나면 reviews/merged.md 를 만들어:
1. 두 리뷰가 모두 지적한 항목 (신뢰도 높음)
2. 한쪽만 지적한 항목 — 각각 출처 URL을 열어서 네가 직접 확인하고 맞음/틀림 표시
3. 서로 충돌하는 지적 — 원문 확인 후 판정
4. report.md에 반영할 최종 수정 목록 (섹션·문장 단위, 우선순위순)
report.md 자체는 아직 수정하지 마.
```

병렬로 안 돌리고 하나씩 보려면 `survey-reviewer-opus 서브에이전트로 report.md를 검토하고 reviews/review-opus.md에 저장해줘` 식으로 따로 호출.

## 병합 후

`reviews/merged.md`의 4번 목록을 보고 `report.md`에 반영. 그 다음 나(Claude 앱)에게 merged.md를 주면 정리 문서(Claude Doc)에 보완 내용을 반영해줄 수 있다.

## 예상 비용·시간

리뷰어가 논문당 원문을 열어보게 해뒀으므로 subagent 하나에 20–40분, 웹 호출 수십 건. `maxTurns: 80`으로 상한을 걸어뒀고, 너무 오래 걸리면 체크리스트 3·4(gap·missing work)만 먼저 돌리도록 태스크를 좁혀도 된다.
