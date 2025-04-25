Market fields:
{
'ticker': 'KXPGATOUR-ZCONO25-WHAL', 
'event_ticker': 'KXPGATOUR-ZCONO25', 
'market_type': 'binary', 
'title': 'Will Whaley/Albertson win the Zurich Classic of New Orleans?', 
'subtitle': '', 
'yes_sub_title': 'Whaley/Albertson', 
'no_sub_title': 'Whaley/Albertson', 
'open_time': '2025-04-24T16:00:00Z', 
'close_time': '2027-04-28T01:00:00Z', 
'expected_expiration_time': '2025-04-28T01:00:00Z', 
'expiration_time': '2027-04-28T01:00:00Z', 
'latest_expiration_time': '2027-04-28T01:00:00Z', 
'settlement_timer_seconds': 60, 
'status': 'initialized', 
'response_price_units': 'usd_cent', 
'notional_value': 100, 
'tick_size': 1, 
'yes_bid': 0, 
'yes_ask': 0, 
'no_bid': 100, 
'no_ask': 100, 
'last_price': 0, 
'previous_yes_bid': 0, 
'previous_yes_ask': 0, 
'previous_price': 0, 
'volume': 0, 
'volume_24h': 0, 
'liquidity': 0, 
'open_interest': 0, 
'result': '', 
'can_close_early': True, 
'expiration_value': '', 
'category': '', 
'risk_limit_cents': 0, 
'strike_type': 'structured', 
'custom_strike': {'golf_competitor': 'aacd0f16-88e0-4b67-8ea1-a01723067201'}, 
'rules_primary': 'If Whaley/Albertson wins the Zurich Classic of New Orleans, then the market resolves to Yes.', 
'rules_secondary': ''
}

IMPORTANT AGGS:
'volume': 0, 
'volume_24h': 0,
'open_interest': 0, 
'liquidity': 0, 

Trades fields:
{
'trade_id': '1b777f28-40c7-4fe1-9e79-b14a1d536625', 
'ticker': 'KXNASDAQ100U-25APR24H1200-T19189.99', 
'count': 100, 
'created_time': '2025-04-24T15:04:34.842737Z', 
'yes_price': 4, 
'no_price': 96, 
'taker_side': 'yes'
}

IMPORTANT AGGS:
'count': 100, * ('yes_price': 4, OR 'no_price': 96, DEPENDING ON 'taker_side': 'yes')

events:
{
'event_ticker': 'KXROBOTMARS-35', 
'series_ticker': 'KXROBOTMARS', 
'sub_title': 'Before 2035', 
'title': 'Will a humanoid robot walk on Mars before a human does?', 
'collateral_return_type': '', 
'mutually_exclusive': False, 
'category': 'Science and Technology'
}

IMPORTANT:
'category': 'Science and Technology'
