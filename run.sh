#!/bin/bash
export PYTHONPATH=$(pwd)
streamlit run src/frontend/🏠_Menu.py  \
    --theme.primaryColor="rgb(137, 180, 250)" \
    --theme.backgroundColor="rgb(244, 219, 214)" \
    --theme.secondaryBackgroundColor="rgb(235, 160, 172)" \
    --theme.textColor="rgb(49, 0, 85)" \
    --theme.base="dark" \
    --theme.font="Georgia, serif" \

