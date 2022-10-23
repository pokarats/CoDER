# -*- coding: utf-8 -*-

import requests
import lxml.html as lh
from lxml.html import fromstring

baseUrl = "https://utslogin.nlm.nih.gov"
authEndpoint = "/cas/v1/api-key"

class Authentication:

   def __init__(self, apiKey):
      self.apiKey = apiKey
      self.service = "http://umlsks.nlm.nih.gov"

   def getTgt(self):
      params = {'apikey': self.apiKey}
      h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent":"python" }
      r = requests.post(baseUrl + authEndpoint, data = params, headers = h)
      response = fromstring(r.text)

      tgt = response.xpath('//form/@action')[0]
      
      return tgt

   def getServiceTicket(self,tgt):

     params = {'service': self.service}
     h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent":"python" }
     r = requests.post(tgt, data = params, headers = h)
     st = r.text
     
     return st
      

