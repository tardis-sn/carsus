""" Object-relational mapping helpers """


class UniqueMixin(object):
    """
    Unique object mixin.

    Allows an object to be returned or created as needed based on
    criterion.

    .. seealso::

        http://www.sqlalchemy.org/trac/wiki/UsageRecipes/UniqueObject

    """
    @classmethod
    def unique_hash(cls, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def unique_filter(cls, query, *args, **kwargs):
        raise NotImplementedError()


    @classmethod
    def as_unique(cls, session, *args, **kwargs):

        hashfunc = cls.unique_hash
        queryfunc = cls.unique_filter
        constructor = cls

        if 'unique_cache' not in session.info:
            session.info['unique_cache'] = cache = {}
        else:
            cache = session.info['unique_cache']

        key = (cls, hashfunc(*args, **kwargs))
        if key in cache:
            return cache[key]
        with session.no_autoflush:
            q = session.query(cls)
            q = queryfunc(q, *args, **kwargs)
            obj = q.first()
            if not obj:
                obj = constructor(*args, **kwargs)
                session.add(obj)
        cache[key] = obj
        return obj


def yield_limit(qry, pk_attr, maxrq=100):
    """Specialized windowed query generator (using LIMIT/OFFSET)
    This recipe is to select through a large number of rows thats too
    large to fetch at once. The technique depends on the primary key
    of the FROM clause being an integer value, and selects items
    using LIMIT.

    The recipe is taken from https://bitbucket.org/zzzeek/sqlalchemy/wiki/UsageRecipes/WindowedRangeQuery
    """

    firstid = None
    while True:
        q = qry
        if firstid is not None:
            q = qry.filter(pk_attr > firstid)
        rec = None
        yield from q.order_by(pk_attr).limit(maxrq)
        if rec is None:
            break
        firstid = pk_attr.__get__(rec, pk_attr) if rec else None


