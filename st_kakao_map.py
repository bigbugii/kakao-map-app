import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not KAKAO_API_KEY or not OPENAI_API_KEY:
    st.error(".env 파일에 KAKAO_API_KEY와 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

HEADERS = {
    "Authorization": f"KakaoAK {KAKAO_API_KEY}"
}

CATEGORY_CODES = {
    "음식점": "FD6",
    "카페": "CE7",
    "편의점": "CS2",
    "주차장": "PK6",
}


def geocode_address(address: str) -> dict:
    """주소 또는 장소명을 좌표로 변환합니다."""
    address_url = "https://dapi.kakao.com/v2/local/search/address.json"

    response = requests.get(
        address_url,
        headers=HEADERS,
        params={"query": address},
        timeout=10,
    )
    response.raise_for_status()

    documents = response.json().get("documents", [])

    # 주소 검색 실패 시 키워드 검색으로 전환
    if not documents:
        keyword_url = "https://dapi.kakao.com/v2/local/search/keyword.json"

        response = requests.get(
            keyword_url,
            headers=HEADERS,
            params={"query": address, "size": 1},
            timeout=10,
        )
        response.raise_for_status()
        documents = response.json().get("documents", [])

    if not documents:
        raise ValueError("주소 또는 장소를 찾을 수 없습니다.")

    document = documents[0]

    return {
        "address": (
            document.get("road_address_name")
            or document.get("address_name")
            or address
        ),
        "x": float(document["x"]),
        "y": float(document["y"]),
    }


def search_nearby_places(
    x: float,
    y: float,
    category_code: str,
    radius: int,
) -> list[dict]:
    """좌표 주변 장소를 가까운 순서로 검색합니다."""
    url = "https://dapi.kakao.com/v2/local/search/category.json"

    params = {
        "category_group_code": category_code,
        "x": x,
        "y": y,
        "radius": radius,
        "sort": "distance",
        "size": 15,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    return response.json().get("documents", [])


def make_dataframe(places: list[dict]) -> pd.DataFrame:
    """카카오 검색 결과를 데이터프레임으로 변환합니다."""
    rows = []

    for place in places:
        rows.append(
            {
                "상호명": place.get("place_name", "이름 없음"),
                "주소": (
                    place.get("road_address_name")
                    or place.get("address_name")
                    or "주소 없음"
                ),
                "거리(m)": int(place["distance"])
                if place.get("distance", "").isdigit()
                else None,
                "전화번호": place.get("phone", ""),
                "위도": float(place["y"]),
                "경도": float(place["x"]),
                "상세 링크": place.get("place_url", ""),
            }
        )

    return pd.DataFrame(rows)


def explain_places(
    location: str,
    category: str,
    places_df: pd.DataFrame,
) -> str:
    """검색 결과를 객관적으로 정리합니다."""
    place_text = places_df[
        ["상호명", "주소", "거리(m)", "전화번호"]
    ].to_string(index=False)

    prompt = f"""
검색 기준 위치: {location}
카테고리: {category}

검색된 장소 목록:
{place_text}

위 장소를 가까운 순서대로 정리해주세요.

조건:
- 맛이나 품질을 평가하거나 추천하지 마세요.
- 장소명, 주소, 거리, 전화번호를 표시하세요.
- 도보 시간은 거리 80m당 약 1분으로 계산하고 추정값이라고 표시하세요.
- 검색 결과에 없는 정보는 추가하지 마세요.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "너는 카카오맵 검색 결과를 객관적으로 정리하는 도우미야. "
            "추천이나 품질 판단은 하지 마."
        ),
        input=prompt,
    )

    return response.output_text


# 세션 상태 초기화
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "places" not in st.session_state:
    st.session_state.places = pd.DataFrame()

if "explanation" not in st.session_state:
    st.session_state.explanation = ""


st.set_page_config(
    page_title="카카오 장소 검색 에이전트",
    page_icon="📍",
    layout="wide",
)

st.title("📍 주소 근처 장소 찾기")
st.caption("카카오맵 API와 OpenAI를 활용한 장소 검색·정리 에이전트")

with st.sidebar:
    st.header("검색 조건")

    location = st.text_input(
        "주소 또는 장소명",
        placeholder="예: 강원대학교",
    )

    category = st.selectbox(
        "카테고리",
        options=list(CATEGORY_CODES.keys()),
    )

    radius = st.slider(
        "검색 반경(m)",
        min_value=100,
        max_value=20000,
        value=1000,
        step=100,
    )

    search_button = st.button(
        "🔍 검색",
        type="primary",
        use_container_width=True,
    )

if search_button:
    if not location.strip():
        st.error("주소 또는 장소명을 입력해주세요.")
        st.stop()

    try:
        with st.spinner("장소를 검색하는 중입니다..."):
            coordinate = geocode_address(location)

            places = search_nearby_places(
                x=coordinate["x"],
                y=coordinate["y"],
                category_code=CATEGORY_CODES[category],
                radius=radius,
            )

        if not places:
            st.error("주변에 검색된 장소가 없습니다.")
            st.stop()

        places_df = make_dataframe(places)

        with st.spinner("검색 결과를 정리하는 중입니다..."):
            explanation = explain_places(
                location,
                category,
                places_df,
            )

        st.session_state.places = places_df
        st.session_state.explanation = explanation
        st.session_state.search_history.append(
            {
                "검색 위치": location,
                "카테고리": category,
                "반경(m)": radius,
                "검색 결과 수": len(places),
            }
        )

        st.success(f"검색 기준 위치: {coordinate['address']}")

    except requests.RequestException as error:
        st.error(f"카카오 API 호출 중 오류가 발생했습니다: {error}")

    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"오류가 발생했습니다: {error}")


places_df = st.session_state.places

if not places_df.empty:
    st.subheader("🗺️ 검색 위치")

    map_df = places_df.rename(
        columns={
            "위도": "lat",
            "경도": "lon",
        }
    )

    st.map(map_df[["lat", "lon"]])

    st.subheader("📋 검색 결과")

    display_df = places_df[
        ["상호명", "주소", "거리(m)", "전화번호", "상세 링크"]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🤖 AI 정리")
    st.info(st.session_state.explanation)


if st.session_state.search_history:
    st.subheader("🕘 지난 검색 기록")

    history_df = pd.DataFrame(st.session_state.search_history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )