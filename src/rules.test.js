import assert from 'node:assert/strict';
import { scan, isCodeNotice } from './rules.js';

const ids = t => scan(t).flags.map(f => f.id).sort();

assert.deepEqual(ids("Hi mom, I'll be home for dinner at seven."), []);
assert.equal(scan('See you tomorrow at the office').score, 0);
assert.deepEqual(ids('Please read me the OTP we just sent'), ['otp']);
assert.deepEqual(ids('Pay with Google Play cards right now'), ['giftcard', 'urgency']);
assert.deepEqual(ids("Install AnyDesk and don't tell anyone"), ['remote', 'secrecy']);

const call = "This is the fraud department from your bank. Your account will be frozen. " +
  "Read me the 6-digit code immediately and transfer your money to a safe account.";
assert.equal(scan(call).score, 100);
assert.ok(ids(call).includes('otp') && ids(call).includes('transfer'));

assert.deepEqual(ids('Pwede po ba kayong mag-send sa GCash ko'), ['giftcard']);

assert.deepEqual(ids('Pulis po ito, aarestuhin kayo ngayon din. Wag nyo po sabihin kahit kanino.'), ['secrecy', 'threat', 'urgency']);
assert.deepEqual(ids('Nanalo ka sa raffle! Padalhan mo kami ng processing fee.'), ['prize', 'transfer']);
assert.deepEqual(ids('Anak, kumain ka na ba? Ingat sa biyahe.'), []);

assert.ok(isCodeNotice('Your OTP for BPI Online is 482913. Do not share this code with anyone, including BPI employees.'));
assert.ok(isCodeNotice('Your GCash authentication code is 551204. Never share your code with anyone.'));
assert.ok(!isCodeNotice('Your OTP is 482913. Do not share it. If you did not request this, call 0917 123 4567.'));
assert.ok(!isCodeNotice('Your OTP is 482913. Never share it. Verify at bpi-secure.xyz'));
assert.ok(!isCodeNotice('Please read me the OTP we just sent, do not share it with anyone else.'));

console.log('rules ok');
