(define-library (scheme base)
    (import (ruschm base))
    (export apply car cdr eqv? eq? cons boolean? char? number? string? symbol? pair? procedure? vector? boolean=? not
        + - * / = < <= > >=
        abs min max sqrt exp ln log sin cos tan asin acos atan atan2 floor ceiling exact floor-quotient floor-remainder newline vector make-vector
        vector-length vector-ref vector-set!
        caar cadr cdar cddr caaar caadr cadar caddr cdaar cdadr cddar cdddr
        list make-list null? append
        map for-each fold-left fold-right
        list-tail list-ref last-pair head atom? equal? list?
    )
    (begin
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

        (define (make-list k fill) (if (> k 0) (cons fill (make-list (- k 1) fill)) '()))

        (define (null? x) (eqv? x '()))


        (define (append . lsts)
        (cond
            ((null? lsts) '())
            ((null? (car lsts)) (apply append (cdr lsts)))
            (else (cons (caar lsts) (apply append (cdar lsts) (cdr lsts))))))


        (define (map proc list)
            (if (pair? list)
                (cons (proc (car list)) (map proc (cdr list)))
                list
            )
        )

        ; (define filter
        ;     (lambda (pred lst)
        ;       (cond ((null? lst) '())
        ;             ((pred (car lst)) (cons (car lst) (filter pred (cdr lst))))
        ;             (else (filterb pred (cdr lst))))))


        (define (for-each proc list)
            (if (pair? list)
                ((lambda () (proc (car list)) (for-each proc (cdr list))))))

        (define (fold-left f init seq)
            (if (null? seq)
                init
                (fold-left f
                           (f (car seq) init)
                           (cdr seq))))

        (define (fold-right f init seq)
            (if (null? seq)
                init
                (f (car seq)
                    (fold-right f init (cdr seq)))))

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
        (if (eq? x '())
            #t
            (if (pair? x)
                (if (list? (cdr x)) #t #f)
                #f)))

    )
)
