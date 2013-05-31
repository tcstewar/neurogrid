import shelve

db = shelve.open('/tmp/neurogrid')
    
            
def make_key(**keys):    
    items=['%s=%s'%(k,v) for k,v in sorted(keys.items())]
    return ','.join(items)
    
def get(**keys):
    v = db.get(make_key(**keys), None)
    return v
    
def set(value, **keys):
    db[make_key(**keys)] = value
        
        
if __name__=='__main__':
    print get(a=1, b=1)
    set(3, a=1, b=1)
    print get(a=1, b=1)
            
