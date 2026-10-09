// Known scam scripts (and normal calls) that live speech is compared against with on-device embeddings.
// type === null marks a normal conversation; nearest-neighbour hits on those pull the score down.
export const PATTERNS = [
  ['Bank OTP scam', 'This is your bank fraud department. Please read me the one-time code we just sent to verify your identity.'],
  ['Bank OTP scam', 'Your account will be blocked unless you confirm the verification code sent to your phone.'],
  ['Bank OTP scam', 'We detected an unauthorized transaction. To cancel it, give me the six digit PIN you received by text.'],
  ['Bank OTP scam', 'For security purposes, I need your card number, expiry date and the three digits at the back.'],
  ['Bank OTP scam', 'There is a suspicious transaction on your account and we will freeze it today. Give me the number we sent to your cell phone.'],
  ['Safe account scam', 'Your savings are at risk. Transfer all your money to this safe account while we investigate.'],
  ['Family emergency scam', 'Hi it is your son, I lost my phone, this is my new number. I need money urgently for the hospital.'],
  ['Family emergency scam', 'Your grandson was in an accident and needs bail money right now. Please do not tell his parents.'],
  ['Family emergency scam', 'Mom I am in trouble, please send me money quickly, I will explain later, do not call my old number.'],
  ['Prize / lottery scam', 'Congratulations, you won a brand new car in our raffle. Pay the release fee to claim your prize.'],
  ['Prize / lottery scam', 'You have been selected as our lucky winner. Send a processing fee by gift card to receive your cash prize.'],
  ['Government impersonation', 'This is the tax office. You owe back taxes and a warrant will be issued for your arrest unless you pay today.'],
  ['Government impersonation', 'This is the police. Your identity was used in a crime. Pay a penalty now or you will be arrested.'],
  ['Government impersonation', 'Your social security number has been suspended due to suspicious activity. Press one to speak to an officer.'],
  ['Tech support scam', 'This is Microsoft support. Your computer has a virus. Please install AnyDesk so I can fix it remotely.'],
  ['Tech support scam', 'We detected hackers on your device. Download this app and give me the access code shown on the screen.'],
  ['Parcel delivery scam', 'Your package is on hold at customs. Pay a small delivery fee through this link to release it.'],
  ['Parcel delivery scam', 'We could not deliver your parcel. Confirm your address and card details to reschedule delivery.'],
  ['Investment scam', 'Invest in our crypto trading platform and get guaranteed thirty percent returns every week.'],
  ['Investment scam', 'Only a few slots left. Send your investment in bitcoin today and double your money in a month.'],
  ['Job offer scam', 'You are hired for an online job. Just pay a training and registration fee before you can start earning.'],
  ['Job offer scam', 'Earn five thousand a day liking videos. First deposit a small amount to unlock your tasks.'],
  ['Romance scam', 'My love, I am stuck abroad and my bank account is frozen. Can you send me money for the plane ticket?'],
  ['Refund scam', 'You were overcharged and are owed a refund. Give me your online banking login so I can process it.'],
  ['Utility cut-off scam', 'Your electricity will be disconnected in one hour unless you pay the overdue bill through this number.'],
  ['SIM / account takeover', 'We are upgrading your SIM card. Please tell me the code you receive so we can activate the new one.'],
  ['Bank OTP scam', 'Taga-bangko po ako. Ibigay nyo po yung OTP na pinadala sa inyo para hindi ma-block ang account nyo.'],
  ['Family emergency scam', 'Ma, bagong number ko ito, nasira yung phone ko. Padalhan mo ako ng pera sa GCash, emergency lang.'],
  ['Prize / lottery scam', 'Nanalo po kayo sa promo! Magbayad lang po ng processing fee para ma-claim ang premyo.'],
  ['Government impersonation', 'Pulis po ito. May kaso kayo at aarestuhin kayo kung hindi kayo magbabayad ngayon din.'],
  ['Parcel delivery scam', 'May package po kayo na naka-hold. Magbayad po ng delivery fee para mai-release.'],
  // Types seen in the Philippine spam SMS dataset (Kaggle, bwandowando).
  ['Online casino / gambling spam', 'Mag-register na at makakuha ng FREE 888 bonus! Deposit 100 at maging 1000. Maglaro na sa aming casino site.'],
  ['Online casino / gambling spam', 'Big wins, small bets! Claim your 300% first deposit bonus and free spins today at our online casino.'],
  ['Rigged game scam', 'Boss, papanalunin kita ng 10k sa laro, basta hati tayo. Mag-cash in ka lang ng 200 at i-message mo ako sa FB.'],
  ['Task / easy money job scam', 'Kumita ng 3,000 araw-araw, 1-2 oras lang na trabaho sa cellphone. Bayad agad sa parehong araw. Mag-apply sa link.'],
  ['Loan scam', 'Need cash? Instant loan approval, walang collateral, low interest. Text mo lang ako para sa processing fee.'],
  [null, 'Hey, are we still on for lunch tomorrow? I will bring the slides for the client meeting.'],
  [null, 'Hi, this is the dental clinic calling to confirm your appointment on Thursday at three in the afternoon.'],
  [null, 'Your order has been shipped and will arrive tomorrow. No action is needed.'],
  [null, 'Can you pick up some milk and bread on your way home tonight?'],
  [null, 'Happy birthday! Hope you have a great day, let us celebrate this weekend.'],
  [null, 'The meeting has moved to Monday morning. Let me know if that works for you.'],
  [null, 'Hi, I am calling from the school. Your daughter forgot her lunch box, you can pick it up at the office.'],
  [null, 'This is a reminder that your library books are due next week.'],
  [null, 'Thanks for your payment. Your receipt has been emailed to you.'],
  [null, 'I am running ten minutes late, the traffic is terrible. Start without me.'],
  [null, 'Hello, how are you? I just wanted to catch up, it has been a while since we talked.'],
  [null, 'Your bank statement for this month is now available in the app.'],
  [null, 'Did you watch the game last night? What a finish!'],
  [null, 'The plumber will come on Saturday morning to fix the kitchen sink.'],
  [null, 'Can you send me the photos from the trip when you have time?'],
  [null, 'Nak, kumain ka na ba? Uuwi ka ba ngayong weekend?'],
  [null, 'Pare, kita tayo mamaya sa mall, alas sais.'],
  [null, 'Ma, nakarating na ako sa opisina. Tatawag ako mamaya.'],
].map(([type, text]) => ({ type, text }));

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

// Vectors are L2-normalised, so dot product = cosine similarity.
// Risk comes from how much closer the text is to the nearest scam script than to the nearest normal conversation.
export function matchPattern(vec, patternVecs) {
  let scam = -1, normal = -1, best = -1;
  patternVecs.forEach((p, i) => {
    const s = dot(vec, p);
    if (!PATTERNS[i].type) normal = Math.max(normal, s);
    else if (s > scam) { scam = s; best = i; }
  });
  const margin = scam - normal;
  // ponytail: calibrated on multilingual-e5-small with held-out EN/TL texts (normal ≤ 0.006, scams ≥ 0.019); recalibrate if the model changes.
  const risk = Math.round(Math.max(0, Math.min(1, (margin - 0.01) / 0.05)) * 100);
  return { type: risk ? PATTERNS[best].type : null, sim: scam, risk };
}
