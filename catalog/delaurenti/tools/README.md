# DeLaurenti 카탈로그 수집 도구

delaurenti.com (NitroSell 기반 스토어)의 카테고리 목록 페이지를 수집해 CSV/Markdown으로 정리하는 스크립트.

```bash
python3 scrape.py https://delaurenti.com/grocery-olive-oil-26/   # 목록 페이지 전부 -> grocery-olive-oil-26.json
python3 detail.py grocery-olive-oil-26.json                       # 상세 페이지에서 설명/태그라인/재고 보강
python3 facets.py grocery-olive-oil-26.json                       # 필터 검색으로 브랜드/산지 매핑 -> *.facets.json
python3 organize.py ../ grocery-olive-oil-26.json                 # CSV + README 생성
```

- `cats.txt`: 사이트 내비게이션에서 뽑은 나머지 카테고리 URL 목록.
- 목록 페이지는 `?page=N`(1부터), 필터 검색 결과는 `&page=N`(0부터)로 페이지가 매겨진다.
- 모든 요청은 `cache/`에 저장되어 재실행 시 네트워크를 다시 쓰지 않는다.
