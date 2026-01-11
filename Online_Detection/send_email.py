# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:22:27 2020

@author: jknutzen
"""
import smtplib
from email.message import EmailMessage

def send_email(condition, text):
    message_string=""
    msg = EmailMessage()
    #msg.set_content('HM10 Diagnostics Overview:\n'+text[0]+text[1]+'\n'+text[2]+text[3]+'\n'+text[4]+text[5]+'\n'+text[6]+text[7]+'\n'+text[8]+text[9]+'\n'+text[10]+text[11]+'\n'+text[12]+text[13]+'\n'+text[14]+text[15]+'\n'+text[16]+text[17])
    for element in text:
        message_string=message_string + element
    msg.set_content(message_string)
    msg['Subject']=condition
    msg['From']="smtp_anon_test@magna.com"
    # msg['To']="julian.knutzen@magna.com", "andy.xin@partner.magna.com"
    msg['To'] ="andy.xin@partner.magna.com"
    try:
        s = smtplib.SMTP(host='smtp-anon.magna.global', port=25, timeout=5)
        print("SMTP host reachable")
        s.send_message(msg)
        s.quit()
    except Exception as e:
        print("SMTP host NOT reachable:", e)
    # s = smtplib.SMTP(host='smtp-anon.magna.global', port=25)
    # s.send_message(msg)
    del msg

if __name__=="__main__":
    send_email("Test",["Test", "this is a test"])
