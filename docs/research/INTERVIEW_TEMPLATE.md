# Reviewer interview: notes template

Copy this file to `docs/research/interview-1.md` and fill it in. **Write no personal data.** That means no customer names, ID numbers or file contents, and the interviewee only by role (for example "onboarding reviewer, payments company, 3 years"). Their name stays off the repo unless they agree in writing.

## Before you start (say this, in your own words)

> I'm building a prototype for a hackathon that helps onboarding staff check merchant documents faster. It doesn't use any real customer data. I'd like to ask you about how you review files today, about 30 minutes. I'll take notes, but I won't record your name or any customer details. Is that OK? Can I quote you anonymously?

- Consent to anonymous quotes: yes / no
- Date, duration, in person / call:
- Role and experience (no name):
- Organisation type (bank, e-wallet, telecom…; no name unless they allow it):

## Questions (DESIGN.md Appendix A)

Ask them in order, but follow interesting answers. Write down **exact phrases** where you can: they make the best evidence.

1. Walk me through the last merchant file you reviewed. Which documents, and in what order?
2. How long does a typical file take? A bad one? What makes it bad?
3. Which fields do you actually check on each document? Which do you re-type?
4. How do you compare names across documents? What counts as "the same name"? (Ask about عبد الله / عبدالله, a missing laqab, گ / ك.)
5. How often do photos come back unreadable? What happens then, and how long does it take?
6. What mistakes have slipped through? What did they cost?
7. Are there handwritten or stamped documents? Which ones?
8. What would make you trust an automatic "checks passed"? What would make you distrust it?
9. What should the screen show you first?
10. Which other documents do you require in practice (residence card, lease, bank letter)?

## If there is time: show the prototype (5 minutes)

Open the reviewer workbench on a fictional packet (`var/demo/3-conflict-injection`) and ask:
- What do you look at first? What is confusing?
- Would the "the OCR read: …" suggestion save you time, or would you ignore it?
- Would you trust the green مطابق fields without opening them?

## Notes

| Question | What they said (exact words where possible) |
|---|---|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |
| 6 | |
| 7 | |
| 8 | |
| 9 | |
| 10 | |
| Prototype | |

## Three things that surprised me

1.
2.
3.

## What we change because of this

-
