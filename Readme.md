## 객체 추출 프로토타입 서버

- 담당자가 개발을 완료하기 전까지 사용할 목적으로 객체 추출 서버를 생성한다.

## 로컬 기동 방법

1. `.env.exemple`을 참고하여 `.env` 파일을 생성한다.
    - `HOST_OUTPUTS_DIR`=`파이썬 서버에 의해 크롭된 이미지가 저장될 경로`
    - `HOST_INPUTS_DIR`=`파이썬 서버로 요청을 보낼 이미지가 있는 경로`
2. `docker-compose up -d`를 통해 도커 컨테이너를 실행한다.

FYI. `docker-compose.yml`을 보면 알겠지만 8989 포트로 요쳥 시 파이썬 서버로 포워드 된다.

## API

- 엔드포인트: `/extracts`
- 메서드: `POST`
- 요청 페이로드:
    - `request_id`: 요청에 대한 id값으로 `string` 타입의 값을 받는다.
- 내용:
    - `request_id`를 기반으로 `HOST_INPUTS_DIR`에서 이미지를 찾는다.
        - `HOST_INPUTS_DIR`가 `/path`이고, `request_id`가 `123`이라면
        - `/path/123.*`인 파일을 모두 탐색 후 첫번째 이미지를 가져온다.
    - `HOST_OUTPUTS_DIR`과 `request_id`를 조합하여 하위 경로에 크롭한 이미지를 저장한다.
        - `HOST_INPUTS_DIR`가 `/path`이고, `request_id`가 `123`이라면
        - `/path/123/name.jpg`로 이미지를 저장한다.
- 반환값
    ```json
    {
        "request_id": "req_20240522_001",
        "count": 2,
        "detected_items": [
            {
                "label": "Bottle",
                "detail": "유리병류",
                "confidence": {
                    "object": 0.813,
                    "detail": 0.953
                },
                "filename": "081c60be.jpg"
            },
            ...
        ]
    }
```