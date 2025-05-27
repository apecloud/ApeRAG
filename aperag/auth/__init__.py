# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from auth0.authentication.token_verifier import AsymmetricSignatureVerifier, TokenVerifier

from config import settings

# https://aperag.jp.auth0.com/.well-known/jwks.json
"""
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "n": "nlKTJh7mDBbQi0PEhgr2wgVvNhQoCsBV6xprWkQThFewy2cvxdBFnKGpA7xHBuM56FJEX0685VoAxz-6LYmgnDwi6ho9_JG7JgfB3F9eA-0dyGj5Jf-af7ZubdOIQahRE3l6sPmv_QaZiFXrGqv4uHrahJLZjHhTtovj7yZ_s-kenDRHuENz4CiXWRpM5SoX32e7EhIL14gD3P94n4fJYVTEhWj9dlLZcVaRKla8JoX8_q-i8TKj7rxggODxAEbnchhbYUTeA_AT6dTEFNo5M5ypjUgP2k2G6mon1yiLc8tD54hkoJYj_DsrteYlV_delIwRgK0j3Qtb3nZHdsCU_w",
      "e": "AQAB",
      "kid": "Brz6M6JbXJJSc69pp1sYm",
      "x5t": "OZHu9BiZZI3JKDzXtYlha8hBwf4",
      "x5c": [
        "MIIDBTCCAe2gAwIBAgIJeQrDBhXOxwNCMA0GCSqGSIb3DQEBCwUAMCAxHjAcBgNVBAMTFWt1YmVjaGF0LmpwLmF1dGgwLmNvbTAeFw0yMzA2MjYwNjIyMDhaFw0zNzAzMDQwNjIyMDhaMCAxHjAcBgNVBAMTFWt1YmVjaGF0LmpwLmF1dGgwLmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAJ5SkyYe5gwW0ItDxIYK9sIFbzYUKArAVesaa1pEE4RXsMtnL8XQRZyhqQO8RwbjOehSRF9OvOVaAMc/ui2JoJw8IuoaPfyRuyYHwdxfXgPtHcho+SX/mn+2bm3TiEGoURN5erD5r/0GmYhV6xqr+Lh62oSS2Yx4U7aL4+8mf7PpHpw0R7hDc+Aol1kaTOUqF99nuxISC9eIA9z/eJ+HyWFUxIVo/XZS2XFWkSpWvCaF/P6vovEyo+68YIDg8QBG53IYW2FE3gPwE+nUxBTaOTOcqY1ID9pNhupqJ9coi3PLQ+eIZKCWI/w7K7XmJVf3XpSMEYCtI90LW952R3bAlP8CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUPRjq9q1qHzAFT7stAFdOn+uUJrowDgYDVR0PAQH/BAQDAgKEMA0GCSqGSIb3DQEBCwUAA4IBAQAOmEXbIkuYP5vwP+isZSf6G1m8jXpHfQtaZUYWBztwwNuEBXbzDgTUH69NnjYg9Hqgkl7g0015DG3HoBuLNdP2hMFJkjzwyvNVyyo9C4gd8mK7r83JlSKYAvfYWdAY5/19g4ock9WdOL3KgPOYK/hXEc26xoWVLkXQ5bs8XKhSalMR7bkAuFNyFxgFgkPsG6alfzYJcLT3V32A9QIcyT5yZ8XI8ZMsshpikAdhVyuz096+biOdLaFHEuvq+tkty/a6qf5A3d7U1mI2iv+ayOQphkAyKZ+mgygzyz3+npD6vgx/5AuTzVaYJYpeIUTULlOobHx3O6EPF7lJsGbqidMK"
      ],
      "alg": "RS256"
    },
    {
      "kty": "RSA",
      "use": "sig",
      "n": "2k71kz9szlLpugj__PO8XQYdJ4NnYRRsMyKS0Th_EFDXy8fnjpFgfwh80zJDmd1gUyrZct7DrjI3weMWYDgHs3BjCcacpNyPEIYp9mmGiBTm9vJMImzmE9rHc2IBpYsIhZGpVCDTAe_8COmqiICDdWosxNtaQ4iGUXKNzAyZ7u5uaTHV93uAbggiUEZJXGHCIv3gdEPR1GhCex6ZTkfbkTef675sQ525113uaPo2ZNCygS2I4jCiEVsoePlcIIjt8Va6u80JpWU0aBZxpjTv3DsyoAfWliuZF6Bafifhnfjvwu1a3J_xwCvFj6OBNXr8ibMyqSWslsYn2f_23hfLXw",
      "e": "AQAB",
      "kid": "Q16ujP_GbuQM8EnEIk1eN",
      "x5t": "UEDdO0p8-eGR4383TTc7qsznLzo",
      "x5c": [
        "MIIDBTCCAe2gAwIBAgIJN0GAUGcDZEV5MA0GCSqGSIb3DQEBCwUAMCAxHjAcBgNVBAMTFWt1YmVjaGF0LmpwLmF1dGgwLmNvbTAeFw0yMzA2MjYwNjIyMDhaFw0zNzAzMDQwNjIyMDhaMCAxHjAcBgNVBAMTFWt1YmVjaGF0LmpwLmF1dGgwLmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANpO9ZM/bM5S6boI//zzvF0GHSeDZ2EUbDMiktE4fxBQ18vH546RYH8IfNMyQ5ndYFMq2XLew64yN8HjFmA4B7NwYwnGnKTcjxCGKfZphogU5vbyTCJs5hPax3NiAaWLCIWRqVQg0wHv/AjpqoiAg3VqLMTbWkOIhlFyjcwMme7ubmkx1fd7gG4IIlBGSVxhwiL94HRD0dRoQnsemU5H25E3n+u+bEOduddd7mj6NmTQsoEtiOIwohFbKHj5XCCI7fFWurvNCaVlNGgWcaY079w7MqAH1pYrmRegWn4n4Z3478LtWtyf8cArxY+jgTV6/ImzMqklrJbGJ9n/9t4Xy18CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUWFOdHbH29MPyfv3iPIzfTDepKFYwDgYDVR0PAQH/BAQDAgKEMA0GCSqGSIb3DQEBCwUAA4IBAQBmvSXbXxAXsrA2jwqyeHx0lHr5HT9OrBmpJCJ5cxxugyDi+FNBlKQnUaAwyi24CaJVCYkaUK/yCxndlAVTnmeJjoyXRjYvLwgNh3xGkKQSDEU5ybWw8Rsk4sLH/MTTtMHwg/Y7e89ZVelf6HFI4tVczef0wIUNV/k6iLdJ/sE7/iUdvQ0c59lgNhkehvbwmEDu0Zf0ZOIzTYVgyP1Ash7Ku1uNMHV4zio3lAUrEpozcsF9Jyd0Pwq86mRICFqOjVSazHaWccQvkHsDbXrLapF43v8rsHqMYNveA7Y56ecBd3uP45a/LrM+abzm+FBG+fcHvM1kEQDe6cP8A2LBLClR"
      ],
      "alg": "RS256"
    }
  ]
}
"""


def get_jwt_token_verifier(jwks_url, issuer, client_id):
    sv = AsymmetricSignatureVerifier(jwks_url)
    return TokenVerifier(signature_verifier=sv, issuer=issuer, audience=client_id)


match settings.AUTH_TYPE:
    case "auth0":
        issuer = f"https://{settings.AUTH0_DOMAIN}/"
        jwks_url = f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json"
        tv = get_jwt_token_verifier(jwks_url, issuer, settings.AUTH0_CLIENT_ID)
    case "authing":
        issuer = f"https://{settings.AUTHING_DOMAIN}/oidc"
        jwks_url = f"https://{settings.AUTHING_DOMAIN}/oidc/.well-known/jwks.json"
        tv = get_jwt_token_verifier(jwks_url, issuer, settings.AUTHING_APP_ID)
    case "logto":
        issuer = f"http://{settings.LOGTO_DOMAIN}/oidc"
        jwks_url = f"http://{settings.LOGTO_DOMAIN}/oidc/jwks"
        tv = get_jwt_token_verifier(jwks_url, issuer, settings.LOGTO_APP_ID)
    case _:
        tv = None
