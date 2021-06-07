(import (scheme base) (scheme write))

(define (fib x)
    (define (fib-internal acc x)
        (if (< x 2)
            (+ acc 1)
            (fib-internal (fib-internal acc (- x 2)) (- x 1))))
    (fib-internal 0 x))

(fib 26)
