(define make-seq-gen (lambda ()
            (define current 0)
            (lambda () (set! current (+ current 1)) current )))

(define seq-gen (make-seq-gen))
(display (seq-gen))
(newline)
(display (seq-gen))
(newline)
(display (seq-gen))
(newline)

