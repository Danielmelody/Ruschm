(import (scheme base) (scheme write))
; (define (fold f init seq)
;             (if (null? seq)
;                 init
;                 (fold f
;                            (f (car seq) init)
;                            (cdr seq))))

(define ones (make-list 100000 1))
(display (fold + 0 ones))
(newline)
