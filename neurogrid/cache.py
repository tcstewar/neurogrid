import shelve

db = shelve.open('/tmp/neurogrid')

def flush():
    global db
    db.close()
    db = shelve.open('/tmp/neurogrid')
    
            
def make_key(**keys):    
    items=['%s=%s'%(k,v) for k,v in sorted(keys.items())]
    return ','.join(items)
    
def get(**keys):
    v = db.get(make_key(**keys), None)
    return v
    
def set(value, **keys):
    db[make_key(**keys)] = value
      
# Context manager for the caching system
class Item:
    def __init__(self, **keys):
        self.key = make_key(**keys)
    
    def __enter__(self):
        return self
        
    def __exit__ (self, type, value, tb):
        pass
        
    def set(self, value):
        db[self.key] = value    
    
    def get(self):
        return db.get(self.key, None)    
        
    def __str__(self):
        return '"%s"' % self.key    
        
        
if __name__=='__main__':

    print get(a=1, b=1)
    set(3, a=1, b=1)
    print get(a=1, b=1)
    
    with Item(c=1, d=3) as item:   # not sure that this is a useful feature
        print item.get()
        item.set(4)
        print item.get()
            
