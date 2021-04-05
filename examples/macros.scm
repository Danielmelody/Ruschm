(import (scheme base) (scheme write))

(display (begin 1 2 1 2 3))
(newline)

(display (cond
    ((< 2 1) => 0)
    ((< 3 2) => 1)
    ((< 4 3) => 2)
    (else 4)))
(newline)

(let ((a 1)(b 2)) (display a)(display b))
(newline)

(define foo 1)

(and (< 2 1) (begin (set! foo 2) #t))
(display "foo is ")
(display foo)
(newline)

(or (< 2 1) (begin (set! foo 2) #t))
(display "foo is ")
(display foo)
(newline)

(let* ((a 1)(b (+ a 2))) (display a)(display b))
(newline)

(define-syntax vector-to-list
    (syntax-rules ()
        ((vector-to-list #(element ...)) '(element ...))))

(display (vector-to-list #(1 2 3)))
(newline)
