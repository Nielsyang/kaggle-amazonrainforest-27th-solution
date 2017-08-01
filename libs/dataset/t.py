while True:
    try:
        a,b = raw_input().split()
        set_a = set(a)
        set_b = set(b)
        set_jiao = set_a & set_b
        
        if sorted(list(set_a)) == sorted(list(set_jiao)) and sorted(list(set_b)) == sorted(list(set_jiao)):
            print('true')
        else:
            print('false')
    except:
        break