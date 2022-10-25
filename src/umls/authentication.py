#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Authenticating and requesting from UMLS API.


@author: Saadullah Amin
"""

import requests
from lxml.html import fromstring

baseUrl = "https://utslogin.nlm.nih.gov"
authEndpoint = "/cas/v1/api-key"


class Authentication:

    def __init__(self, api_key):
        self.apiKey = api_key
        self.service = "http://umlsks.nlm.nih.gov"

    def get_tgt(self):
        params = {'apikey': self.apiKey}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(baseUrl + authEndpoint, data=params, headers=h)
        response = fromstring(r.text)

        tgt = response.xpath('//form/@action')[0]

        return tgt

    def get_service_ticket(self, tgt):
        params = {'service': self.service}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(tgt, data=params, headers=h)
        st = r.text

        return st
