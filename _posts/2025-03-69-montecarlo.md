---
layout: post
title: "How good is good enough?"
date: 2025-03-15
---
# Decidability and Tractability, Polynomial Hierarchy
I think computer science is pretty neat. What I find neat about it is that a correct algorithm never screws up, whether you do it on one "thing" or a billion of them. When I took a semester of linear algebra last year, I had pretty good intuiton about vector spaces (maybe two semesters of physics weren't such a waste after all). A matrix maps one vector space to another, and if you can invert that matrix, you can undo that mapping. I can sit back in my chair, close my eyes, and "see" the matrix spinning and flipping and squishing vectors. Give me five vectors, and I can "see" if they're perpendicular.

However, I still bombed my tests and finished with a C+ in the course. That's because I would always screw up things like Gauss-Jordan elimination-- I understand exactly how it works, but I'd always say some nonsense like saying 4 x 4 = 12. Just incorrectly doing the algorithm. I'm not a computer, so doing algorithms isn't my strong suit.

But what I think is the neatest thing about computer science is that if you perform the algorithm correctly (which computers usually do), you can solve the problem of any size. You remember phone books? If you want to look for someone in the phone book, you just open to the middle, and if they're before the names on the page, you repeat that process with the stack of pages on the left side, and if they're after, you do it with the stack on the right. That's a fundamental algorithm. I'll call it phonebook search.

Let's think about how long it'd take a computer to phonebook search through as many names as there's **elementary particles in the entire universe**. There's about 10^90 of those. Now let's assume a computer can compare two names in about a billionth of a second-- an M3 MacBook Pro's clock runs at about four gigahertz, which is around four operations in a billionth of a second, so we're in that ballpark. 

In the phonebook search, after one iteration, we've cut the amount of names to look through down by half. So after two, it's 1/2 times 1/2, which you know the answer to. After 10, it's 1/2 raised to the tenth power. So, to phonebook search through a given amount of things, it'll take the base-2 logarithm amount of iterations **in the worst case**.

So, stay with me, we can compare two names in a billionth of a second. The base-2 logarithm of 10^90 is a little under 300, so we have to do around 300 iterations of the phonebook search algorithm. That's 300 nanoseconds. A laptop (not a supercomputer. A laptop.) can search through a phonebook containing **as many names as there are protons, neutrons, electrons, and other things in the entire universe in less than 3% of a millisecond**.

It just so happens that looking through the phonebook is a problem that there's a **fast** algorithm to solve. There are a lot of important problems that we **don't think** there could **ever possibly be** an algorithm to solve that's both fast and always correct. This is called the P vs NP problem-- it's an open question that asks "does us being able to check a solution quickly imply that it's possible to solve it just as quickly?" We don't think so, because there are a lot (thousands) of problems from business, to finance, to healthcare, to engineering, to games like Candy Crush or sudoku that we can mathematically prove are equally difficult to one another, so if we can solve one, we can solve them all, but we haven't found a fast, always-correct algorithm for a single one. Seriously, there are a lot of things that're equally hard:

- Given a chain of amino acids, determine the shape of the protein they naturally fold themselves into. (We could probably cure a lot of diseases if we could do this quickly.)
- Given a bunch of resources (let's say barrels of wheat, barley, and hops, water, electricity, and warehouse space for a brewery), determine how much of our budget we should spend on each to maximize profit. (This is important for business.)
- Given a description of what an electrical circuit should do, make it as small as possible. (This is important for making fast computers as cheaply as possible.)
- Given a Candy Crush or Sudoku game, win. (This isn't important.)

The encryption that makes the internet secure is the opposite of this-- it relies on those types of problems being basically impossible to solve. The best algorithms we know to **always correctly** solve these problems are **literally brute-force guess-and-check**. There's a million-dollar bounty on an algorithm faster than that. (And a Stanford professorship waiting for you.) Also, if you solved it, you could crack the Bitcoin blockchain, and send every single bitcoin that exists to yourself, which is much more than a million dollars.

The simplest "hard" problem is called 3-SAT. It asks, given some formula of a bunch of boolean (they're either true or false) variables, in the form `(var1 OR var2 OR NOT var2) AND (NOT var2 OR var5 OR NOT var13) ...`, there's some true/false assignments of all those variables that make the whole formula evaluate to `true`. For example, if we have the formula `(x OR NOT y OR NOT z) AND (x OR y OR z)` there's at least one satisfying assignment: if we set `x`, `y`, and `z` to true, the whole formula evaluates to true. The output is just that singular true/false value, of whether or not there's at least a singular satisfying assignment of those variables.

3-SAT obviously tells you nothing useful-- it's less useful than Candy Crush, which at least gives you dopamine from the flashing colors. However, there are a lot of problems across domains that "boil down to" 3-SAT, so there are a lot of commercial programs that **usually** solve gigantic 3-SAT instances really fast, using an algorithm that's **always correct**. What they all have in common is that there's some input that could make them take forever.

When I say forever, I mean forever. The phonebook search takes time proportional to the logarithm of the number of names you've got, but that's kind of unfair to compare with 3-SAT-- logarithms really cut down the size of a number. We'll sort Uno cards instead, which you also do with an algorithm which takes time proportional to the **square** of the amount of cards you've got (much, much slower than phonebook search). Your computer can sort 1000 Uno cards in around a thousandth of a second.

Back to 3-SAT. In the worst case, if you've got a 3-SAT instance with 1000 variables, and you can check the formula in a nanosecond, your computer could take on the order of 10^284 **years**. The heat death of the universe is around 10^100 years from now. Those are both algorithms that do *something* with 1000 *things*, but a thousandth of a second is pretty different than "until after the heat death of the universe.

As I said before, solving 3-SAT is just guess-and-check (with some incredibly sophisticated "rules of thumb," in commercial solvers). You try every variable being set to false, and check if that's correct. If it isn't, you set the first to true, and the rest to false, and check that. Then you set only the second to true, and the rest to false, and check that... If you haven't found the satisfying assignment, repeat this process until you've set them all to true, and **only then** can you say **for sure** that the formula is or isn't satisfiable.

Count how many tries you could need for two variables, then three variables, then four... you'll see 4, then 8, then 16, then 32 for five variables, then 64 for six... the time this guess-and-check takes is proportional to 2 (because there are two choices for each variable-- true or false) **raised to the number of variables you have**. And that's assuming you can instantly check whether or not that assignment actually satisfies the formula, because I'm in a good mood today.

# Chernoff Bounds and Monte Carlo Algorithms
If you took an exam, and you wanted a 100, and you got a 99.999999999% (that's nine nines after the decimal place) because your professor didn't like your handwriting, or something, you'd probably be fine with that. In the same way, if I could solve 3-SAT really quickly with that accuracy, that'd be pretty cool. So **I'll do that right now: I'll give you an algorithm to solve 3-SAT to 99.999999999% (the same nine nines after the decimal place) accuracy in the blink of an eye**.

You ready? Here's the algorithm. What we'll do is randomly assign every variable to true or false, and then check if the whole formula evaluates to true. If it does, say that (obviously), and if it doesn't, flip a coin-- heads, true, tails, false.

Obviously, that doesn't have anywhere near the 99.999999999% chance of correctness I promised. It's closer to a 50.001% chance of correctness. If that was a midterm, you failed, and the amount of 0s between the 5 and the 1 doesn't change that. You failed that midterm pretty fast, though.

What we're actually going to do is do that "guess" algorithm repeatedly until we're bored (for a fixed number of iterations). If it didn't work, find a new random assignment, and check again. Over and over.

Recall that the brute-force solution could take 10^284 years, in the worst case. Let's say we're doing that 1000-variable formula, our laptop can check if our assignment satisfies the formula in a thousandth of a second, and we had a 0.001% chance of getting it right with the random assignment, just to make our numbers easy to play with. That means our algorithm took a thousandth of a second and has a 50.001% chance of correctness and a 49.999% chance of whatever the opposite of that is. Not very good.

Now let's try it three times, with new random assignments, and take the majority vote of the outcomes. That takes 0.003 seconds, and has a 49.999%^3 chance of being incorrect. Run the numbers-- that's a 12% chance of being incorrect. We're at an 88% on our exam-- we just went from an F to a B+. Pretty good.

Now, let's run it for just a second. That's a thousand runs, and we have a 49.999%^1000 chance of being incorrect. Run those numbers: that's better than a 0.000...0001% chance of incorrectness, with **300 zeros**. One second, for 300 nines-worth of accuracy, versus until after the heat death of the universe. That's a worthwhile tradeoff.

Modern cryptography-- what keeps your passwords secure-- is based on this "probably correct" type of algorithm. Are you alright with that accuracy? The entirety of the world's financial systems sure are.

Also, pat yourself on the back! This "talk" was actually a proof of my favorite result from my theory of computation class (Pitt CS 2110): NP is in PP. That's a PhD class about theoretical computer science. And you understand everything perfectly! Awesome stuff!
