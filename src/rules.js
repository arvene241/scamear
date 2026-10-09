// Instant, deterministic scam red flags (English + common Tagalog/Taglish phrasing). Runs on every transcript update; the LLM adds judgment on top.
export const RULES = [
  { id: 'otp', weight: 45, label: 'Asked for a one-time code (OTP)', tip: 'No real bank or company ever asks for your OTP.',
    re: /\b(otp|one[- ]time (pin|code|password)|verification code|6[- ]digit code|code (we|i) (just )?sent|pin code|pinadala(ng)? code)\b/i },
  { id: 'credentials', weight: 40, label: 'Asked for password / PIN / card details', tip: 'Never share passwords, PINs, CVV or full card numbers.',
    re: /\b(password|passcode|cvv|security code|card number|pin number|login details|mpin)\b/i },
  { id: 'giftcard', weight: 45, label: 'Payment by gift card / crypto / e-wallet', tip: 'Gift cards and crypto are untraceable — the #1 scam payment.',
    re: /\b(gift ?cards?|google play cards?|itunes|steam cards?|bitcoin|crypto|usdt|e-?load|western union|remittance|g-?cash|maya|palawan|cebuana)\b/i },
  { id: 'remote', weight: 45, label: 'Wants remote access to your device', tip: 'Never install AnyDesk/TeamViewer for a caller.',
    re: /\b(any ?desk|team ?viewer|remote access|screen ?share|install (this|an?) app|download (this|an?) app)\b/i },
  { id: 'urgency', weight: 20, label: 'Pressure / urgency', tip: 'Scammers rush you so you can\'t think. Slow down.',
    re: /\b(right now|immediately|urgent(ly)?|within (the next )?\d+ (minutes|hours)|today only|last chance|act fast|before it'?s too late|ngayon din|bilisan|agad-agad|madalian)\b/i },
  { id: 'threat', weight: 30, label: 'Threat of arrest, fines or account closure', tip: 'Police and agencies don\'t threaten arrest over the phone.',
    re: /\b(arrest(ed)?|warrant|police|lawsuit|jail|penalt(y|ies)|suspend(ed)?|blocked|frozen|deactivated?|legal action|pulis|aarestuhin|kakasuhan|makukulong|ipapakulong|multa)\b/i },
  { id: 'secrecy', weight: 30, label: 'Told to keep it secret', tip: 'Being told "don\'t tell anyone" is a giant red flag.',
    re: /\b(don'?t tell (anyone|anybody|your)|keep (this|it) (a )?secret|confidential|don'?t hang up|stay on the line|(wag|huwag) (mo|mong|nyo|niyo|ninyong) (po )?(na )?(sabihin|ipaalam))\b/i },
  { id: 'prize', weight: 25, label: 'Too-good-to-be-true prize or refund', tip: 'You can\'t win a contest you never entered.',
    re: /\b(you('ve| have)? won|winner|prize|lottery|jackpot|refund|cash ?back|claim your|congratulations|nanalo|premyo)\b/i },
  { id: 'transfer', weight: 30, label: 'Asked to move or send money', tip: 'Real banks never ask you to move money to a "safe account".',
    re: /\b(safe account|transfer (the |your )?(money|funds)|send (the )?money|wire (the )?money|processing fee|release fee|pay (a|the) fee|magpadala ng pera|padalhan)\b/i },
  { id: 'impersonation', weight: 15, label: 'Claims to be bank / government / courier', tip: 'Hang up and call the official number yourself.',
    re: /\b(from (your|the) bank|fraud department|bank officer|irs|bir|sss|customs|courier|delivery (fee|failed)|tech(nical)? (support|department)|microsoft|amazon|taga-?bangko|bdo|bpi|landbank|nbi|philhealth)\b/i },
];

// A legit one-time-code notice (banks, GCash, Shopee send these daily): it *gives* a code, warns not to share it,
// and has no link or phone number to act on. Phishing copies almost always add a link or a number to call.
export function isCodeNotice(text) {
  return /\b(otp|code|pin)\b[^.]{0,40}?\b(is|:)\s*\d{4,8}\b/i.test(text)
    && /\b(do not|don'?t|never|huwag|wag)\b[^.]{0,20}\b(share|ibigay|sabihin)/i.test(text)
    && !/https?:\/\/|www\.|\b[a-z0-9-]+\.(com|ph|net|xyz|site|tv|top|icu|link|ly)\b|(\+63|\b09)\d{2}[\s-]?\d{3}/i.test(text);
}

// Returns matched rules (each at most once) and a 0-100 risk score.
export function scan(text) {
  const flags = RULES.filter(r => r.re.test(text));
  const score = Math.min(100, flags.reduce((s, r) => s + r.weight, 0));
  return { flags, score };
}
