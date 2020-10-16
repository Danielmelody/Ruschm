;These functions come mostly from [minischeme](https://github.com/catseye/minischeme)

(define (caar x) (car (car x)))
(define (cadr x) (car (cdr x)))
(define (cdar x) (cdr (car x)))
(define (cddr x) (cdr (cdr x)))
(define (caaar x) (car (car (car x))))
(define (caadr x) (car (car (cdr x))))
(define (cadar x) (car (cdr (car x))))
(define (caddr x) (car (cdr (cdr x))))
(define (cdaar x) (cdr (car (car x))))
(define (cdadr x) (cdr (car (cdr x))))
(define (cddar x) (cdr (cdr (car x))))
(define (cdddr x) (cdr (cdr (cdr x))))

(define (list . x) x)

(define (null? x) (eqv? x '()))

(define (map proc list)
    (if (pair? list)
        (cons (proc (car list)) (map proc (cdr list)))
        list
    )
)


; (define (for-each proc list)
;     (if (pair? list)
;         (begin (proc (car list)) (for-each proc (cdr list)))
;         #t ))

(define (list-tail x k)
    (if (= k 0)
        x
        (list-tail (cdr x) (- k 1))))

(define (list-ref x k)
    (car (list-tail x k)))

(define (last-pair x)
    (if (pair? (cdr x))
        (last-pair (cdr x))
        x))

(define (head stream) (car stream))

;;;;	atom?
(define (atom? x)
  (not (pair? x)))

; `cond` requires macro system
;;;;	memq
; (define (memq obj lst)
;   (cond
;     ((null? lst) #f)
;     ((eq? obj (car lst)) lst)
;     (else (memq obj (cdr lst)))))

;;;;    equal?
(define (equal? x y)
  (if (pair? x)
    (and (pair? y)
         (equal? (car x) (car y))
         (equal? (cdr x) (cdr y)))
    (and (not (pair? y))
         (eqv? x y))))

(define (list? x)
  (or (eq? x '())
      (and (pair? x)
           (list? (cdr x)))))
