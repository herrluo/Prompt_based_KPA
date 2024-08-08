import streamlit as st

st.set_page_config(
    page_title="Thesis Demo",
    page_icon="🎈",
    layout="wide"
)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with st.expander("About this demo", expanded=True):
    st.markdown("")
    st.write(
        """     
    The Demo will show the following:
-   Extracting nested key points out of multiple datasets. 
-   Matching the key points to sentences in text.
        """
    )

st.markdown("")
st.markdown("")
st.markdown("")

main_tab1, main_tab2, main_tab3 = st.tabs(["KPG Results", "KPM Results", "CTG Live"])

# Content for KPG Results tab
with main_tab1:
    st.markdown("")
    st.markdown("")
    c1, c2 = st.columns([2, 4])

    with c1:
        tab1_topic = st.selectbox(
            "Topic",
            [
                "Assisted suicide should be a criminal offence",
                "Homeschooling should be banned",
                "The vow of celibacy should be abandoned",
                "We should abandon marriage",
                "We should abandon the use of school uniform",
                "We should abolish capital punishment",
                "We should abolish intellectual property rights",
                "We should abolish the right to keep and bear arms",
                "We should adopt an austerity regime",
                "We should adopt atheism",
                "We should adopt libertarianism",
                "We should ban human cloning",
                "We should ban private military companies",
                "We should ban the use of child actors",
                "We should close Guantanamo Bay detention camp",
                "We should end affirmative action",
                "We should end mandatory retirement",
                "We should fight for the abolition of nuclear weapons",
                "We should fight urbanization",
                "We should introduce compulsory voting",
                "We should legalize cannabis",
                "We should legalize prostitution",
                "We should legalize sex selection",
                "We should prohibit flag burning",
                "We should prohibit women in combat",
                "We should subsidize journalism",
                "We should subsidize space exploration",
                "We should subsidize vocational education",
                "Routine child vaccinations should be mandatory",
                "Social media platforms should be regulated by the government",
                "The USA is a good country to live in"
            ],
            help="At present, you can choose one topic"
        )

        tab1_stance = st.selectbox(
            "Stance",
            ["Positive", "Negative"],
            help="Choose a stance. ",
        )

        tab1_llm = st.selectbox(
            "LLM",
            ["GPT4", "LLama2", "Qwen"],
            help="Select the LLM.",
            key='tab1_llm'
        )

        tab1_methods = st.selectbox(
            "KPG Method",
            ["Prompt-based Direct in Group Generation", "Prompt-based Hybrid in 1by1 Generation"],
            help="Select the KPG method.",
            key='tab1_kpg'
        )

    with c2:
        st.markdown("**Topic:**  \n" + tab1_topic)
        st.markdown("**Stance:**  \n" + tab1_stance)
        text = ' '
        sample_output = [
            "Assisted suicide could potentially open the door for individuals with ulterior motives to coerce or manipulate vulnerable patients into ending their lives.",
            "Assisted suicide contradicts the Hippocratic oath taken by healthcare professionals to preserve life and do no harm.",
            "Assisted suicide is essentially aiding in the death of another person, which can be seen as equivalent to murder and should therefore be treated as a criminal offense.",
            "The possibility of a cure or the advent of new treatment options in the future could render the act of assisted suicide premature and unnecessary.",
            "Assisted suicide could inadvertently put pressure on the terminally ill to end their life out of fear of being a burden to their loved ones."
        ]
        text += f'**<p> Key Points: </p><ol>**'
        for s in sample_output:
            text += "<li>" + s + "</li>"
        text += "</ol>"
        st.markdown(text, unsafe_allow_html=True)

# Content for KPM Results tab
with main_tab2:
    st.markdown("")
    st.markdown("")
    c1, c2 = st.columns([2, 4])

    with c1:
        tab2_llm = st.selectbox(
            "LLM",
            ["GPT4", "LLama2", "Qwen"],
            help="Select the LLM.",
            key='tab2_llm'
        )

        tab2_kpg = st.selectbox(
            "KPG Method",
            ["Prompt-based Direct in Group Generation", "Prompt-based Hybrid in 1by1 Generation"],
            help="Select the KPG method.",
            key='tab2_kpg'
        )

        tab2_kpm = st.selectbox(
            "KPM Method",
            ["Group-based Direct Method", "Group-based Hybrid Method", "1by1-based Hybrid Method"],
            help="Select the KPM method.",
        )

    with c2:
        image_path = "./image/gpt4_kpm.png"
        st.image(image_path, caption="gpt4")

# Content for CTG Live tab
with main_tab3:
    st.markdown("")
    st.markdown("")
    c1, c2 = st.columns([2, 4])

    with c1:
        tab3_technology = st.selectbox(
            "Technique",
            ["Prompt-based CTG", "CtrlSumm with Local Ranking"],
            help="Choose a CTG Technique. ",
        )

        tab3_llm = st.selectbox(
            "LLM",
            ["GPT4", "LLama2", "Qwen"],
            help="Select the LLM.",
            key='tab3_llm'
        )
        if tab3_technology == "CtrlSumm with Local Ranking":
            tab3_kpg = st.selectbox(
                "KPG Method",
                ["Prompt-based Direct in Group Generation", "Prompt-based Hybrid in 1by1 Generation"],
                help="Select the KPG method.",
                key='tab3_kpg'
            )

    with c2:
        input_text = st.text_area('Enter text here:')
        key_word = st.text_area('Enter keywords here:')
        submit_button = st.button(label="✨ Show me !")
        text = ' '
        sample_output = [
            "The USA is ranked among the world's worst places to move to, due to rapidly increasing costs of living, healthcare, and basic education.",
            "The USA doesn't provide free healthcare, has high crime rates, racism, xenophobia, and high tax rates.",
            "The USA has a significant gap between the rich and poor, and the poorest in society lack good healthcare and an adequate benefits system.",
            "The USA's culture promotes materialism and forces unbridled consumerism, leading to a stressful pace of life and an endless search for money and status.",
            "The USA, although having a good health sector, is criticized for its excessive costs, inequality, strict immigration rules, and its culture of war."
        ]
        text += f'**<p> Key Points: </p><ol>**'
        for s in sample_output:
            text += "<li>" + s + "</li>"
        text += "</ol>"
        if submit_button:
            st.markdown(text, unsafe_allow_html=True)
