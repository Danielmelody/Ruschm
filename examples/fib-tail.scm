(define (fib-internal acc x)
    (if (< x 2)
        (+ acc 1)
        (fib-internal (fib-internal acc (- x 2)) (- x 1))
    )
)
(define (fib x) 
    (fib-internal 0 x))

(define (fib-seq-internal i x)
    (display (fib i))
    (newline)
    (if (< i x)
        (fib-seq-internal (+ i 1) x)
        1
    )
)

(define fib-seq
    (lambda x
        (fib-seq-internal 0 x)))

(fib-seq 26)