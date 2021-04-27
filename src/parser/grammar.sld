(define-syntax begin
    (syntax-rules ()
        ((begin exp1 ...  )
            ((lambda () exp1 ... )))))

(define-syntax let
    (syntax-rules ()
        ((let ((name val) ...) body ...)
            ((lambda (name ...) body ...)
                val ...))))

(define-syntax let*
    (syntax-rules ()
        ((let* () body ...)
            (let () body ...))
        ((let* ((name val))
               body ...)
            (let ((name val))
                 body ...))
        ((let* ((name1 val1) (name2 val2) ...)
               body ...)
            (let ((name1 val1))
                (let* ((name2 val2) ...)
                 body ...)))))


(define-syntax cond
      (syntax-rules (else =>)
        ((cond (else result ...))
         (begin result ...))
        ((cond (test => result))
         (let ((temp test))
           (if temp (result temp))))
        ((cond (test => result) clause ...)
         (let ((temp test))
           (if temp
               (result temp)
               (cond clause ...))))
        ((cond (test)) test)
        ((cond (test) clause ...)
         (let ((temp test))
            (if temp
                temp
               (cond clause ...))))
        ((cond (test result ...))
         (if test (begin result ...)))
        ((cond (test result ...)
               clause ...)
         (if test
             (begin result ...)
             (cond clause ...)))))

(define-syntax case
     (syntax-rules (else =>)
       ((case (key ...)
          clauses ...)
        (let ((atom-key (key ...)))
          (case atom-key clauses ...)))
       ((case key
           (else => result))
           (result key))
       ((case key
           (else result ...))
           (begin result ...))
       ((case key
           ((atoms ...) => result))
           (if (not (null? (memv key '(atoms ...))))
              (result key)))
       ((case key
          ((atoms ...) result ...))
        (if (memv key '(atoms ...))
            (begin result ...)))
       ((case key
           ((atoms ...) => result)
               clauses ...)
           (if (memv key '(atoms ...))
               (result key)
               (case key clauses ...)))
       ((case key
           ((atoms ...) result ...)
               clauses ...)
           (if (memv key '(atoms ...))
               (begin result ...)
               (case key clauses ...)))))

(define-syntax and
      (syntax-rules ()
        ((and) #t)
        ((and test) test)
        ((and test1 test2 ...)
         (if test1 (and test2 ...) #f))))

(define-syntax or
      (syntax-rules ()
        ((or) #f)
        ((or test) test)
        ((or test1 test2 ...)
         (let ((x test1))
           (if x x (or test2 ...))))))

(define-syntax when
      (syntax-rules ()
        ((when test result1 result2 ...)
         (if test
             (begin result1 result2 ...)))))

(define-syntax unless
      (syntax-rules ()
        ((unless test result1 result2 ...)
         (if (not test)
             (begin result1 result2 ...)))))
