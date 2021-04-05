(import (scheme base) (scheme write))

(define-syntax macro1
    (syntax-rules ()
        ((macro1 exp1 ...  )
            (+ exp1 ... ))))

(define-syntax macro2
    (syntax-rules (test)
        ((macro2 test exp1 ...  )
            (+ exp1 ... ))))

(macro1 1 2)
(macro2 test 1 2)

