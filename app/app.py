"""
app/app.py  -  Step 7: Streamlit web app
Run from the project root:  streamlit run app/app.py
(The light theme is set in .streamlit/config.toml, so keep that folder.)
"""
import os
import sys
import streamlit as st

# Let Python find the files inside src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

from features import digit_ratio, name_equals_username      # text -> numbers
from predict import predict_profile, format_percent         # the trained model

# ---------------- Colours (one place, easy to change) ----------------
INK, SLATE = "#16212E", "#5B6B7C"
TEAL, BRICK, OCHRE = "#0F766E", "#B42318", "#9A5B00"

st.set_page_config(page_title="Fake Profile Check", page_icon="🔍", layout="centered")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] {{ font-family: 'Public Sans', 'Segoe UI', Arial, sans-serif; }}
[data-testid="stSidebar"] {{ border-right: 1px solid #DDE3EA; }}
h1.page-title {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2.2rem; margin: 0 0 .3rem 0; color: {INK}; }}
p.page-sub {{ color: {SLATE}; margin: 0 0 1.5rem 0; max-width: 62ch; line-height: 1.55; }}
.verdict {{ border-left: 8px solid var(--c); background: #fff; padding: 1.1rem 1.3rem; border-radius: 6px;
           box-shadow: 0 1px 3px rgba(22,33,46,.10); }}
.verdict .word {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2.3rem; line-height: 1.15; color: var(--c); }}
.verdict .meta {{ color: {SLATE}; margin-top: .3rem; }}
.gauge {{ margin: 1.6rem 0 .4rem 0; }}
.gauge .track {{ position: relative; height: 16px; border-radius: 8px;
                background: linear-gradient(90deg, {TEAL} 0%, {TEAL} 35%, #D9DEE4 35%, #D9DEE4 65%, {BRICK} 65%, {BRICK} 100%); }}
.gauge .pin {{ position: absolute; top: -6px; width: 4px; height: 28px; background: {INK}; border-radius: 2px;
              transform: translateX(-2px); }}
.gauge .pinlabel {{ position: relative; height: 1.5rem; font-size: .85rem; font-weight: 600; color: {INK}; }}
.gauge .pinlabel span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}
.gauge .ends {{ display: flex; justify-content: space-between; font-size: .8rem; color: {SLATE}; margin-top: .5rem; }}
.gauge .zone {{ text-align: center; font-size: .78rem; color: {SLATE}; margin-top: .2rem; }}
.disclaimer {{ border-top: 1px solid #DDE3EA; margin-top: 2rem; padding-top: .8rem; font-size: .85rem; color: {SLATE}; }}
</style>
"""

ADVICE = {
    "FAKE": [
        "Do not click links in the bio or send them personal details, one-time codes or money.",
        "Do not accept their requests or reply to their messages.",
        "Use Instagram's **Report** option, then block the account.",
        "If they claim to be someone you know, contact that person on another channel (phone call or another app).",
    ],
    "REAL": [
        "The profile looks normal, but a real-looking account can still be hacked or run by a scammer.",
        "Never share passwords, one-time codes or bank details in chat, even with someone who looks genuine.",
        "Be extra careful with offers that seem too good to be true, or requests for money.",
    ],
    "BORDERLINE": [
        "Treat this account with caution until you can check it another way.",
        "Look at the content yourself: are the photos original, and do the comments come from real people?",
        "Avoid sharing personal details or clicking links until you are sure.",
        "If it claims to be someone you know, confirm with them on another channel.",
    ],
}

# ---------------- Input defaults and ready-made examples ----------------
DEFAULTS = {"f_username": "", "f_fullname": "", "f_pic": "Yes", "f_bio_text": "",
            "f_bio_skip": False, "f_bio_size": "No bio", "f_url": "No",
            "f_private": "No", "f_posts": 0, "f_followers": 0, "f_following": 0}

EXAMPLES = {
    "Fake-looking": {"f_username": "user48291834", "f_fullname": "", "f_pic": "No",
                     "f_bio_text": "", "f_bio_skip": False, "f_bio_size": "No bio",
                     "f_url": "No", "f_private": "No", "f_posts": 0, "f_followers": 8, "f_following": 30},
    "Real-looking": {"f_username": "priya.sharma", "f_fullname": "Priya Sharma", "f_pic": "Yes",
                     "f_bio_text": "Coffee lover | Sharing photos, places, and everyday moments!",
                     "f_bio_skip": False, "f_bio_size": "No bio",
                     "f_url": "Yes", "f_private": "No", "f_posts": 180, "f_followers": 900, "f_following": 400},
    "Borderline": {"f_username": "alex_3030", "f_fullname": "Alex Rao", "f_pic": "Yes",
                   "f_bio_text": "Just here!", "f_bio_skip": False, "f_bio_size": "No bio",
                   "f_url": "No", "f_private": "No", "f_posts": 15, "f_followers": 150, "f_following": 100},
}


def get_bio_length(text: str, skip: bool, size_label: str) -> int:
    """Count pasted bio text, or use the selected size when the text is skipped."""
    if not skip:
        return len(text)

    size_values = {
        "No bio": 0,
        "Short (one line)": 20,
        "Medium (2-3 lines)": 60,
        "Long (full bio)": 120,
    }
    return size_values[size_label]


def load_example(name):
    """Fill the sidebar with an example and check it straight away."""
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value
    st.session_state["auto_check"] = True


def build_profile(has_pic, username, fullname, bio_length, has_url, is_private, posts, followers, following):
    """Turn the sidebar answers into the dictionary that predict_profile() expects."""
    return {
        "profile_pic": int(has_pic),
        "username_digit_ratio": digit_ratio(username),               # calculated from text
        "fullname_words": len(fullname.split()),                     # calculated from text
        "fullname_digit_ratio": digit_ratio(fullname),               # calculated from text
        "name_equals_username": name_equals_username(fullname, username),  # calculated from text
        "bio_length": int(bio_length),
        "external_url": int(has_url),
        "private": int(is_private),
        "posts": int(posts),
        "followers": int(followers),
        "following": int(following),
    }


def verdict_style(result):
    """Pick colour, headline and advice group (borderline results get their own wording)."""
    if result["borderline"]:
        return OCHRE, "Not sure", "BORDERLINE"
    if result["label"] == "FAKE":
        return BRICK, "Likely FAKE", "FAKE"
    return TEAL, "Likely REAL", "REAL"


def gauge_html(p_fake):
    """Confidence bar: the pin shows where the model places the profile between REAL and FAKE."""
    pos = min(max(p_fake * 100, 3), 97)
    return (
        f'<div class="gauge" role="img" aria-label="Fake probability {format_percent(p_fake)}">'
        f'<div class="pinlabel"><span style="left:{pos}%">{format_percent(p_fake)} chance fake</span></div>'
        f'<div class="track"><div class="pin" style="left:{pos}%"></div></div>'
        f'<div class="ends"><span>Looks like a real account</span><span>Looks like a fake account</span></div>'
        f'<div class="zone">The grey middle (35% to 65%) is the zone where the model is unsure.</div>'
        f'</div>'
    )


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.markdown('<h1 class="page-title">Fake profile check</h1>'
                '<p class="page-sub">Enter what you can see on an Instagram profile. The tool compares it with '
                'patterns learned from 4,934 labelled accounts and tells you how suspicious it looks.</p>',
                unsafe_allow_html=True)

    # ---------------- Sidebar: examples + profile details ----------------
    with st.sidebar:
        st.subheader("Profile details")
        st.caption("No profile to hand? Load an example and the check runs straight away.")
        cols = st.columns(3)
        for col, name in zip(cols, EXAMPLES):
            col.button(name, on_click=load_example, args=(name,), use_container_width=True)
        st.divider()

        st.text_input("Username", key="f_username", placeholder="e.g. travel_lover_8841",
                      help="Exactly as shown on the profile, without the @.")
        st.text_input("Full name", key="f_fullname", placeholder="Leave empty if none",
                      help="The display name shown under the picture.")
        st.radio("Profile picture", ["Yes", "No"], key="f_pic", horizontal=True,
                 help="Choose No if it shows the default grey silhouette.")
        st.checkbox("I don't want to paste it", key="f_bio_skip")
        if st.session_state["f_bio_skip"]:
            st.selectbox("Choose a bio size", ["No bio", "Short (one line)", "Medium (2-3 lines)",
                                               "Long (full bio)"], key="f_bio_size")
        else:
            bio_text = st.text_area(
                "Paste the bio here (leave empty if there is no bio)",
                key="f_bio_text",
                max_chars=150,
                help="The text is only counted to calculate its length and is never stored.",
            )
            bio_count = len(bio_text)
            st.caption(f"Counted: {bio_count} characters" if bio_count else "No bio")
        st.radio("Link in bio", ["No", "Yes"], key="f_url", horizontal=True)
        st.radio("Private account", ["No", "Yes"], key="f_private", horizontal=True)
        st.number_input("Posts", min_value=0, step=1, key="f_posts")
        st.number_input("Followers", min_value=0, step=1, key="f_followers")
        st.number_input("Following", min_value=0, step=1, key="f_following")
        button_clicked = st.button("Check profile", type="primary", use_container_width=True)

    s = st.session_state
    submitted = button_clicked or st.session_state.pop("auto_check", False)

    # ---------------- Main area ----------------
    if not submitted:
        st.markdown("**How to use it**")
        st.markdown("1. Enter the details you can see on the profile, using the sidebar on the left.\n"
                    "2. Select **Check profile**.\n"
                    "3. Read the verdict, the reasons behind it and the safety advice.")
        st.caption("The tool looks at: profile picture, username and name pattern, bio, link, privacy setting, "
                   "posts, followers and following. It never looks at photos or messages.")
    elif not s["f_username"].strip():
        st.error("Enter a username so the tool can check how it is written.")
    else:
        bio_length = get_bio_length(s["f_bio_text"], s["f_bio_skip"], s["f_bio_size"])
        profile = build_profile(s["f_pic"] == "Yes", s["f_username"], s["f_fullname"], bio_length,
                                s["f_url"] == "Yes", s["f_private"] == "Yes",
                                s["f_posts"], s["f_followers"], s["f_following"])
        result = predict_profile(profile)
        colour, headline, advice_key = verdict_style(result)

        # Verdict badge + confidence bar
        st.markdown(
            f'<div class="verdict" style="--c:{colour}"><div class="word">{headline}</div>'
            f'<div class="meta">Model confidence in this answer: <b>{format_percent(result["confidence"])}</b>'
            f' &nbsp;(prediction: {result["label"]})</div></div>',
            unsafe_allow_html=True)
        st.markdown(gauge_html(result["fake_probability"]), unsafe_allow_html=True)

        # Why?
        st.subheader("Why?")
        if result["reasons"]:
            intro = ("These details pushed the score towards fake:" if result["label"] == "FAKE"
                     else "These details pushed the score towards real:")
            st.write(intro)
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.write("No single detail stood out; the score comes from many small signals together.")
        st.caption("Method: each detail is swapped for a typical value of the opposite type, "
                   "and we measure how much the score moves. This is a simple what-if explanation.")

        with st.expander("Values the app calculated from your text"):
            st.write(f"- Digits in username: {profile['username_digit_ratio']:.0%}")
            st.write(f"- Words in full name: {profile['fullname_words']}")
            st.write(f"- Digits in full name: {profile['fullname_digit_ratio']:.0%}")
            st.write(f"- Full name same as username: {'yes' if profile['name_equals_username'] else 'no'}")
            st.write(f"- Bio length used: {profile['bio_length']} characters")

        # Safe-usage advice
        st.subheader("What to do next")
        for line in ADVICE[advice_key]:
            st.markdown(f"- {line}")

    st.markdown(
        '<div class="disclaimer"><b>This is a prediction, not proof.</b> The model can be wrong in both '
        'directions: real people can look suspicious and convincing fakes can look normal. '
        'Use the result as a warning sign, never as the only reason to accuse, report or ban someone.</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
