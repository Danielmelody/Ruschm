(import (scheme base) (scheme write))
(define (fib x)
    (if (< x 2)
        1
        (+ (fib (- x 2)) (fib (- x 1)))))

        ;test

(define fib-seq
    (lambda (x)
        (if (< x 1) 1
            (fib-seq (- x 1)))
        (display (fib x))
        (newline)
        1))

(fib-seq 26)
