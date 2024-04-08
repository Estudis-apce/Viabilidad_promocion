
import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['abc']).generate()
print(hashed_passwords)