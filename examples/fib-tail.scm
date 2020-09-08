(define (fib-internal x y i j)
    (if (< i j)
        (fib-internal y (+ x y) (+ i 1) j)
        x
    )
)
(define (fib z) 
    (fib-internal 1 1 0 z))

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

(fib-seq 44)