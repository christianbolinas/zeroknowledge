---
layout: post
title: "Book Review: Algorithmica"
date: 2025-03-06
---
Sorry for the lack of writing! I've been busy writing. I've been working on my compiler textbooklet and my 
fiction book. The first is kind of like Crafting Interpreters, but even simpler, because I strongly feel 
that compilers aren't that complicated (until you get into the theory behind optimization. Then that shit's
black magic).

I read a lot. I'm in a book club (because, at 24, raging face with the actives in your old frat is really, 
really weird, even if you're still in the same town as your chapter). Fiction books are okay. I haven't really
read a mind-blowing one since Skippy Dies. 

The main things I read are computer science textbooks, because I'm a massive nerd. I've made a routine out of
reading them before bedtime. I'd watch YouTube videos instead, but I'm too ADHD to sit and pay attention to 
some talking head yap for an entire ten-minute video. My favorite is Introduction to Algorithms (CLRS). 
Algorithms, A Creative Approach is almost as good. I've read some great ones on other algorithm-related things, 
like Crafting Interpreters, to learn compiler basics and Haskell From First Principles, to learn about the theory
behind functional programming, but in general, I like reading about algorithms the most. 

The issue is that CLRS is the big daddy end boss of algorithms books, and I've already read it, front to back,
three times (After I took data structures, after I took undergrad algorithms, and after I took grad algorithms.
I told you I was a massive nerd), and I understand, like, 45% of it now. So I've been looking all over the 
internet for more algorithms content. 

I finally found a good one. It's called Algorithmica. It's written by a Russian fellow by the name of Sergey
Slotin. You can read it for free on the internet.

Before I start the review, I feel like I should mention that I'm writing this on emacs. (I was using Neovim,
but that's too mainstream), and I'm listening to a live recording of Wretched World by Converge, with Chelsea Wolfe on
vocals right now. I've also started writing in cursive again, so people know that I went to Catholic school. I'm not
sure what the intended takeaways of that are, but I feel like you should know that regardless.

## Summary
I'm one of those spacey, head-in-the-clouds types of dudes who likes thinking really hard about pointless
theoretical shit for its own sake. I'll try and avoid that here, because this is already a review of a 
textbook that I read for fun. I'll do my best to only touch on important things that might have practical 
takeaways. 

The premise behind Algorithmica is that asymptotic complexity analysis-- Akra-Bazzi and all that fun stuff--
isn't always true in practial applications. An extreme example is Python-- bubble sort in even Java will dog 
walk a Python quicksort implementation. This book focuses on the design, implementation, and analysis of 
algorithms in a practical, HPC-focused sense, because in real life, SIMD and other ways to effectively use
hardware outweigh doodling recursion trees and algebraic manipulations of Akra-Bazzi intagrals. It does so 
without any theoretically-unsound "this algorithm is better because the benchmark says so," or any outrageous
levels of background knowledge of modern computer architecture beyond what I had to learn in my mandatory
second-semester comp arch class, or any number theory beyond what you learned in discrete math.

Over the decades, chips (CPUs, but I try to avoid jargon) have been able to do more and more instructions per 
second. Until recently-- we've tried to make them even faster, and they've started melting. We can still fit
more "things" on each chip, though-- we can make CPUs do more "stuff" by telling them to do one instruction
on multiple "things." It's important to keep this in mind as we design algorithms-- if we have to do a bunch
of things, and we can design an algorithm to do four things in one instruction, then it's just as imperative
to do that as it is to use n log n over n^2. 

Designing cache-friendly data structures is equally important for the same reason. (This is also something I've 
observed-- a vector-based structure beat the piss out of a BST in benchmarks, even if linear runtime is 
asymptotically worse than logarithmic.) 

Designing algorithms that don't cause your CPU branch predictor to screw up is again just as important as the prior two
items.

The author provides specific benchmarks-- for example, CPython multiplying two 1024x1024 real-valued matrices takes over 
10 minutes, Java takes 10 seconds, and C with GCC recklessly optimizing everything it possibly can takes just
over half a second. (This isn't to dogpile Python-- `np.dot` takes just over a tenth of a second.) 

The author also teaches you how to effectively optimize empirical performance, in another section, and does a number
of case studies wherein he and other collaborators were able to out-perform the C++ STL implementations in common
tasks. All the requisite number theory (and other math) background is covered in the book-- a sophomore could read it.

Overall, this book can be summed up as 

- understand how OSs work
- understand how compilers work
- understand how CPUs work
- use SIMD
- optimize for cache hits 

## How Good Was It?
This is a good book. I think that computer science classes are often split up into two camps (at least, they are at
my school, and we have a pretty standard curriculum)-- theory classes, where you learn about algorithms and analyze them 
with whiteboards, and systems classes, which aren't theory-heavy, do care about empirical runtimes, and don't have
complicated algorithms (I haven't found Floyd-Warshall in a systems class, anyway). In real life, though, an algorithm's
stopwatch time does matter, and there isn't much focus on that in school. This book does a great job bridging the gap
from theoretical algorithms topics to practical systems topics, and does a good job analyzing how good knowledge from
both of those fields will increase your programs' performance.
