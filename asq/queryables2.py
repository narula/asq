'''Classes which support the Queryable interface.'''

# Copyright (c) 2011 Robert Smallshire.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = 'Robert Smallshire'

import heapq
from itertools import islice
import operator
from asq.selectors import make_selector

from .selectors import identity
from .extension import extend
from ._types import (is_iterable, is_type)
from ._portability import (imap, ifilter, irange, izip, izip_longest,
                          fold, is_callable, OrderedDict, has_unicode_type,
                          itervalues, iteritems, totally_ordered)

# A sentinel singleton used to identify default argument values.
default = object()

class OutOfRangeError(ValueError):
    '''A subclass of ValueError for signalling out of range values.'''
    pass

class Collection(object):
    def __init__(self, name, iterable, query=None):
        self.dependencies = []  # Things who are dependent on me
        self.iterable = iterable
        self.name = name
        if query is not None:
            self.query = query

    def __iter__(self):
        return iter(self.iterable)

    def _iter(self):
        '''Return an unsorted iterator over the iterable.

        Useful in subclasses to obtain a raw iterator over the iterable where
        __iter__ has been overridden.
        '''
        return iter(self.iterable)

    def add_dependency(self, collection):
        self.dependencies.append(collection)

    def insert(self, row):
        self.iterable.append(row)
        for dep in self.dependencies:
            dep.receive_dep_insert(self, row)

    def receive_dep_insert(self, child_collection, dep_row):
        """ Receive an insert from a collection on which I depend.
        Apply it if necessary"""
        if self.query:
            self.query.apply_new_row(self, dep_row)

class Operator(object):
    def __init__(self):
        self.parent = None
        self.child = None
        self.opstr = None

    def __str__(self):
        return self.opstr

class Root(Operator):
    def __init__(self):
        self.parent = None
        self.child = None
        self.opstr = "Root"
    
    def execute(self, input):
        return input

class Select(Operator):
    def __init__(self, selector):
        self.selector = make_selector(selector)
        self.parent = None
        self.child = None
        self.opstr = "Select"

    def execute(self, input):
        return imap(self.selector, input)

class Where(Operator):
    def __init__(self, predicate):
        self.predicate = predicate
        self.parent = None
        self.child = None
        self.opstr = "Where"

    def execute(self, input):
        return ifilter(self.predicate, input)

class GroupBy(Operator):
    def __init__(self, key_selector=identity, element_selector=identity, result_selector=lambda key, grouping: {key : grouping}):
        self.key_selector = key_selector
        self.element_selector = element_selector
        self.result_selector = result_selector
        self.parent = None
        self.child = None
        self.opstr = "GroupBy"

    def execute(self, input):
        key_value_pairs = imap(lambda item: (self.key_selector(item), self.element_selector(item)), input)
        lookup = Lookup(key_value_pairs)
        for key in lookup:
            yield self.result_selector(key, lookup[key])

class Count(Operator):
    def __init__(self):
        self.parent = None
        self.child = None
        self.opstr = "Count"

    def execute(self, input):
        return len(list(input))

class Limit(Operator):
    def __init__(self, num):
        self.num = num
        self.parent = None
        self.child = None

    def execute(self, input):
        return islice(input, self.num)


class Queryable(object):

    def apply_new_row(self, collection, row):
        # test to see if row passes predicate
        # test to see if row passes order by
        # apply to group by based on row key and aggregate (sum, count, function)
        if self.count and self.count_predicate(row):
            collection.incr(row, self.count)
        elif self.sum and self.where(row):
            collection.sum(row, self.sum.field)
        elif self.where(row):
            collection.insert(row)

    def __init__(self, collection):
        '''Construct a Queryable from a Collection'''

        self._iterable = collection
        self.root = Root()
        self.leaf = self.root

    def add_op(self, operator):
        operator.child = self.root.child
        if self.root.child:
            self.root.child.parent = operator
        self.root.child = operator
        operator.parent = self.root
        print "Adding: ", operator, "parent:", operator.parent, "child:", operator.child
        if self.leaf is self.root:
            self.leaf = operator
        assert self.root.parent == None

    def tree(self):
        out = self._iterable.name
        node = self.leaf
        while node is not None:
            out += "->" + node.__str__()
            node = node.parent
        return out

    def execute(self):
        node = self.root
        while node.child is not None:
            node = node.child
        intermediate = self._iterable
        result = intermediate
        while node is not None:
            print "Executing ", node
            result = node.execute(intermediate)
            intermediate = result
            node = node.parent
        return result

    def __iter__(self):
        '''Support for the iterator protocol.

        Allows Queryable instances to be used anywhere an iterable is required.

        Returns:
            An iterator over the values in the query result.

        Raises:
            ValueError: If the Queryable has been closed().
        '''
        if self.closed():
            raise ValueError("Attempt to use closed() Queryable")

        return self.execute()

    def _iter(self):
        '''Return an unsorted iterator over the iterable.

        Useful in subclasses to obtain a raw iterator over the iterable where
        __iter__ has been overridden.
        '''
        return iter(self._iterable)

    def __enter__(self):
        '''Support for the context manager protocol.'''
        return self

    def __exit__(self, type, value, traceback):
        '''Support for the context manager protocol.

        Ensures that close() is called on the Queryable.
        '''
        self.close()
        return False

    def closed(self):
        '''Determine whether the Queryable has been closed.

        Returns:
            True if closed, otherwise False.
        '''
        return self._iterable is None

    def close(self):
        '''Closes the queryable.

        The Queryable should not be used following a call to close. This method
        is idempotent. Other calls to a Queryable following close() will raise
        ValueError.
        '''
        self._iterable = None

    def count(self):
        self.add_op(Count())
        return self

    def select(self, selector):
        '''Transforms each element of a sequence into a new form.

        Each element of the source is transformed through a selector function
        to produce a corresponding element in teh result sequence.

        If the selector is identity the method will return self.

        Note: This method uses deferred execution.

        Args:
            selector: A unary function mapping a value in the source sequence
                to the corresponding value in the generated generated sequence.
                The single positional argument to the selector function is the
                element value.  The return value of the selector function
                should be the corresponding element of the result sequence.

        Returns:
            A Queryable over generated sequence whose elements are the result
            of invoking the selector function on each element of the source
            sequence.

        Raises:
            ValueError: If this Queryable has been closed.
            TypeError: If selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call select() on a closed Queryable.")

        self.add_op(Select(selector))

    def group_by(self, key_selector=identity,
                 element_selector=identity,
                 result_selector=lambda key, grouping: {key : grouping}):
        '''Groups the elements according to the value of a key extracted by a
        selector function.

        Note: This method has different behaviour to itertools.groupby in the
            Python standard library because it aggregates all items with the
            same key, rather than returning groups of consecutive items of the
            same key.

        Note: This method uses deferred execution, but consumption of a single
            result will lead to evaluation of the whole source sequence.

        Args:
            key_selector: An optional unary function used to extract a key from
                each element in the source sequence. The default is the
                identity function.

            element_selector: A optional unary function to map elements in the
                source sequence to elements in a resulting Grouping. The
                default is the identity function.

            result_selector: An optional binary function to create a result
                from each group. The first positional argument is the key
                identifying the group. The second argument is a Grouping object
                containing the members of the group. The default is a function
                which returns a dictionary of key : grouping.

        Returns:
            A sequence of elements of the where each element
            represents a group.  If the default result_selector is relied upon
            this is a list of dictionaries.

        Raises:
            ValueError: If the Queryable is closed().
            TypeError: If key_selector is not callable.
            TypeError: If element_selector is not callable.
            TypeError: If result_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call select_with_index() on a closed "
                             "Queryable.")

        if not is_callable(key_selector):
            raise TypeError("group_by() parameter key_selector={0} is not "
                            "callable".format(repr(key_selector)))

        if not is_callable(element_selector):
            raise TypeError("group_by() parameter element_selector={0} is not "
                            "callable".format(repr(element_selector)))

        if not is_callable(result_selector):
            raise TypeError("group_by() parameter result_selector={0} is not "
                            "callable".format(repr(result_selector)))

        self.add_op(GroupBy(key_selector, element_selector, result_selector))
        return self

    def where(self, predicate):
        '''Filters elements according to whether they match a predicate.

        Note: This method uses deferred execution.

        Args:
            predicate: A unary function which is applied to each element in the
                source sequence. Source elements for which the predicate
                returns True will be present in the result.

        Returns:
            A Queryable over those elements of the source sequence for which
            the predicate is True.

        Raises:
            ValueError: If the Queryable is closed.
            TypeError: If the predicate is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call where() on a closed Queryable.")

        if not is_callable(predicate):
            raise TypeError("where() parameter predicate={predicate} is not "
                                  "callable".format(predicate=repr(predicate)))
        self.add_op(Where(predicate))
        return self

    def limit(self, count=1):
        '''Returns a specified number of elements from the start of a sequence.

        If the source sequence contains fewer elements than requested only the
        available elements will be returned and no exception will be raised.

        Note: This method uses deferred execution.

        Args:
            count: An optional number of elements to take. The default is one.

        Returns:
            A Queryable over the first count elements of the source sequence,
            or the all elements of elements in the source, whichever is fewer.

        Raises:
            ValueError: If the Queryable is closed()
        '''
        if self.closed():
            raise ValueError("Attempt to call take() on a closed Queryable.")

        count = max(0, count)

        self.add_op(Limit(count))
        return self

    def to_list(self):
        '''Convert the source sequence to a list.

        Note: This method uses immediate execution.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_list() on a closed Queryable.")

        # Maybe use with closable(self) construct to achieve this.
        if isinstance(self._iterable, list):
            return self._iterable
        lst = list(self)
        # Ideally we would close here. Why can't we - what is the problem?
        #self.close()
        return lst

    def to_tuple(self):
        '''Convert the source sequence to a tuple.

        Note: This method uses immediate execution.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_tuple() on a closed Queryable.")

        if isinstance(self._iterable, tuple):
            return self._iterable
        tup = tuple(self)
        # Ideally we would close here
        #self.close()
        return tup

    def to_set(self):
        '''Convert the source sequence to a set.

        Note: This method uses immediate execution.

        Raises:
            ValueError: If duplicate keys are in the projected source sequence.
            ValueError: If the Queryable is closed().
        '''
        if self.closed():
            raise ValueError("Attempt to call to_set() on a closed Queryable.")

        if isinstance(self._iterable, set):
            return self._iterable
        s = set()
        for item in self:
            if item in s:
                raise ValueError("Duplicate item value {0} in sequence "
                    "during to_set()".format(repr(item)))
            s.add(item)
        # Ideally we would close here
        #self.close()
        return s

    def to_dictionary(self, key_selector=identity, value_selector=identity):
        '''Build a dictionary from the source sequence.

        Note: This method uses immediate execution.

        Raises:
            ValueError: If the Queryable is closed.
            ValueError: If duplicate keys are in the projected source sequence.
            TypeError: If key_selector is not callable.
            TypeError: If value_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_dictionary() on a closed Queryable.")

        if not is_callable(key_selector):
            raise TypeError("to_dictionary() parameter key_selector={key_selector} is not callable".format(
                    key_selector=repr(key_selector)))

        if not is_callable(value_selector):
            raise TypeError("to_dictionary() parameter value_selector={value_selector} is not callable".format(
                    value_selector=repr(value_selector)))

        dictionary = {}
        for key, value in self.select(lambda x: (key_selector(x), value_selector(x))):
            if key in dictionary:
                raise ValueError("Duplicate key value {key} in sequence during to_dictionary()".format(key=repr(key)))
            dictionary[key] = value
        return dictionary

    def to_str(self, separator=''):
        '''Build a string from the source sequence.

        The elements of the query result will each coerced to a string and then
        the resulting strings concatenated to return a single string. This
        allows the natural processing of character sequences as strings. An
        optional separator which will be inserted between each item may be
        specified.

        Note: this method uses immediate execution.

        Args:
            separator: An optional separator which will be coerced to a string
                and inserted between each source item in the resulting string.

        Returns:
            A single string which is the result of stringifying each element
            and concatenating the results into a single string.

        Raises:
            TypeError: If any element cannot be coerced to a string.
            TypeError: If the separator cannot be coerced to a string.
            ValueError: If the Queryable is closed.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_str() on a closed Queryable.")

        return str(self.__dict__)

    def sequence_equal(self, second_iterable, equality_comparer=operator.eq):
        '''
        Determine whether two sequences are equal by elementwise comparison.

        Sequence equality is defined as the two sequences being equal length
        and corresponding elements being equal as determined by the equality
        comparer.

        Note: This method uses immediate execution.

        Args:
            second_iterable: The sequence which will be compared with the
                source sequence.

            equality_comparer: An optional binary predicate function which is
                used to compare corresponding elements. Should return True if
                the elements are equal, otherwise False.  The default equality
                comparer is operator.eq which calls __eq__ on elements of the
                source sequence with the corresponding element of the second
                sequence as a parameter.

        Returns:
            True if the sequences are equal, otherwise False.

        Raises:
            ValueError: If the Queryable is closed.
            TypeError: If second_iterable is not in fact iterable.
            TypeError: If equality_comparer is not callable.

        '''
        if self.closed():
            raise ValueError("Attempt to call to_tuple() on a closed Queryable.")

        if not is_iterable(second_iterable):
            raise TypeError("Cannot compute sequence_equal() with second_iterable of non-iterable {type}".format(
                    type=str(type(second_iterable))[7: -1]))

        if not is_callable(equality_comparer):
            raise TypeError("aggregate() parameter equality_comparer={equality_comparer} is not callable".format(
                    equality_comparer=repr(equality_comparer)))

        # Try to check the lengths directly as an optimization
        try:
            if len(self._iterable) != len(second_iterable):
                return False
        except TypeError:
            pass

        sentinel = object()
        for first, second in izip_longest(self, second_iterable, fillvalue=sentinel):
            if first is sentinel or second is sentinel:
                return False
            if not equality_comparer(first, second):
                return False
        return True

    def __eq__(self, rhs):
        '''Determine value equality with another iterable.

        Args:
           rhs: Any iterable collection.

        Returns:
            True if the sequences are equal in value, otherwise False.
        '''
        return self.sequence_equal(rhs)

    def __ne__(self, rhs):
        '''Determine value inequality with another iterable.

        Args:
           rhs: Any iterable collection.

        Returns:
            True if the sequences are inequal in value, otherwise False.
        '''
        return not (self == rhs)


    # Methods for more Pythonic usage

    # Note: __len__ cannot be efficiently implemented in an idempotent fashion
    # (without consuming the iterable or changing the state of the object. Call
    # count() instead see
    # http://stackoverflow.com/questions/3723337/listy-behavior-is-wrong-on-first-call
    # for more details. This is problematic if a Queryable is consumed using the
    # list() constructor, which calls __len__ prior to constructing the list as
    # an efficiency optimisation.

    def __contains__(self, item):
        '''Support for membership testing using the 'in' operator.

        Args:
            item: The item for which to test membership.

        Returns:
            True if item is in the sequence, otherwise False.
        '''

        return self.contains(item)

    def __getitem__(self, index):
        '''Support for indexing into the sequence using square brackets.

        Equivalent to element_at().

        Args:
            index: The index should be between zero and count() - 1 inclusive.
                Negative indices are not interpreted in the same way they are
                for built-in lists, and are considered out-of-range.

        Returns:
            The value of the element at offset index into the sequence.

        Raises:
            ValueError: If the Queryable is closed().
            IndexError: If the index is out-of-range.
        '''
        try:
            return self.element_at(index)
        except OutOfRangeError as e:
            raise IndexError(str(e))

    def __reversed__(self):
        '''Support for sequence reversal using the reversed() built-in.

        Called by reversed() to implement reverse iteration.

        Equivalent to the reverse() method.

        Returns:
            A Queryable over the reversed sequence.

        Raises:
            ValueError: If the Queryable is closed().
        '''
        return self.reverse()

    def __repr__(self):
        '''Returns a stringified representation of the Queryable.

        The string will *not* necessarily contain the sequence data.

        Returns:
            A stringified representation of the Queryable.
        '''
        # Must be careful not to consume the iterable here
        return 'Queryable({iterable})'.format(iterable=self._iterable)

    def __str__(self):
        '''Returns a stringified representation of the Queryable.

        The string *will* necessarily contain the sequence data.

        Returns:
            A stringified representation of the Queryable.
        '''
        return self.to_str()

if has_unicode_type():

    @extend(Queryable)
    def __unicode__(self):
        '''Returns a stringified unicode representation of the Queryable.

        Note: This method is only available on Python implementations which
            support the named unicode type (e.g. Python 2.x).

        The string *will* necessarily contain the sequence data.

        Returns:
            A stringified unicode representation of the Queryable.
        '''
        return self.to_unicode()

    @extend(Queryable)
    def to_unicode(self, separator=''):
        '''Build a unicode string from the source sequence.

        Note: This method is only available on Python implementations which
            support the named unicode type (e.g. Python 2.x).

        The elements of the query result will each coerced to a unicode
        string and then the resulting strings concatenated to return a
        single string. This allows the natural processing of character
        sequences as strings. An optional separator which will be inserted
        between each item may be specified.

        Note: this method uses immediate execution.

        Args:
            separator: An optional separator which will be coerced to a
                unicode string and inserted between each source item in the
                resulting string.

        Returns:
            A single unicode string which is the result of stringifying each
            element and concatenating the results into a single string.

        Raises:
            TypeError: If any element cannot be coerced to a string.
            TypeError: If the separator cannot be coerced to a string.
            ValueError: If the Queryable is closed.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_unicode() on a closed "
                             "Queryable.")

        return unicode(separator).join(self.select(unicode))


class OrderedQueryable(Queryable):
    '''A Queryable representing an ordered iterable.

    The sorting implemented by this class is an incremental partial sort so
    you don't pay for sorting results which are never enumerated.'''

    def __init__(self, iterable, order, func):
        '''Create an OrderedIterable.

            Args:
                iterable: The iterable sequence to be ordered.
                order: +1 for ascending, -1 for descending.
                func: The function to select the sorting key.
        '''
        assert abs(order) == 1, 'order argument must be +1 or -1'
        super(OrderedQueryable, self).__init__(iterable)
        self._funcs = [(order, func)]

    def then_by(self, key_selector=identity):
        '''Introduce subsequent ordering to the sequence with an optional key.

        The returned sequence will be sorted in ascending order by the
        selected key.

        Note: This method uses deferred execution.

        Args:
            key_selector: A unary function the only positional argument to
                which is the element value from which the key will be
                selected.  The return value should be the key from that
                element.

        Returns:
            An OrderedQueryable over the sorted items.

        Raises:
            ValueError: If the OrderedQueryable is closed().
            TypeError: If key_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call then_by() on a "
                             "closed OrderedQueryable.")

        if not is_callable(key_selector):
            raise TypeError("then_by() parameter key_selector={key_selector} "
                    "is not callable".format(key_selector=repr(key_selector)))

        self._funcs.append((-1, key_selector))
        return self

    def then_by_descending(self, key_selector=identity):
        '''Introduce subsequent ordering to the sequence with an optional key.

        The returned sequence will be sorted in descending order by the
        selected key.

        Note: This method uses deferred execution.

        Args:
            key_selector: A unary function the only positional argument to
                which is the element value from which the key will be
                selected.  The return value should be the key from that
                element.

        Returns:
            An OrderedQueryable over the sorted items.

        Raises:
            ValueError: If the OrderedQueryable is closed().
            TypeError: If key_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call then_by() on a closed OrderedQueryable.")

        if not is_callable(key_selector):
            raise TypeError("then_by_descending() parameter key_selector={key_selector} is not callable".format(key_selector=repr(key_selector)))

        self._funcs.append((+1, key_selector))
        return self

    def __iter__(self):
        '''Support for the iterator protocol.

        Returns:
            An iterator object over the sorted elements.
        '''

        # Determine which sorting algorithms to use
        directions = [direction for direction, _ in self._funcs]
        direction_total = sum(directions)
        if direction_total == -len(self._funcs):
            # Uniform ascending sort - do nothing
            MultiKey = tuple

        elif direction_total == len(self._funcs):
            # Uniform descending sort - invert sense of operators
            @totally_ordered
            class MultiKey(object):
                def __init__(self, t):
                    self.t = tuple(t)

                def __lt__(lhs, rhs):
                    # Uniform descending sort - swap the comparison operators
                    return lhs.t > rhs.t

                def __eq__(lhs, rhs):
                    return lhs.t == rhs.t
        else:
            # Mixed ascending/descending sort - override all operators
            @totally_ordered
            class MultiKey(object):
                def __init__(self, t):
                    self.t = tuple(t)

                # TODO: [asq 1.1] We could use some runtime code generation here to compile a custom comparison operator
                def __lt__(lhs, rhs):
                    for direction, lhs_element, rhs_element in zip(directions, lhs.t, rhs.t):
                        cmp = (lhs_element > rhs_element) - (rhs_element > lhs_element)
                        if cmp == direction:
                            return True
                        if cmp == -direction:
                            return False
                    return False

                def __eq__(lhs, rhs):
                    return lhs.t == rhs.t

        # Uniform ascending sort - decorate, sort, undecorate using tuple element
        def create_key(index, item):
            return MultiKey(func(item) for _, func in self._funcs)

        lst = [(create_key(index, item), index, item) for index, item in enumerate(self._iterable)]
        heapq.heapify(lst)
        while lst:
            key, index, item = heapq.heappop(lst)
            yield item


class Lookup(object):
    '''A multi-valued dictionary.

    A Lookup represents a collection of keys, each one of which is mapped to
    one or more values. The keys in the Lookup are maintained in the order in
    which they were added. The values for each key are also maintained in
    order.

    Note: Lookup objects are immutable.
    '''

    def __init__(self, key_value_pairs):
        '''Construct a Lookup with a sequence of (key, value) tuples.

        Args:
            key_value_pairs:
                An iterable over 2-tuples each containing a key, value pair.
        '''
        # Maintain an ordered dictionary of groups represented as lists
        self._dict = OrderedDict()
        for key, value in key_value_pairs:
            if key not in self._dict:
                self._dict[key] = []
            self._dict[key].append(value)

    def _iter(self):
        return itervalues(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key):
        '''The sequence corresponding to a given key, or an empty sequence if
           there are no values corresponding to that key.

        Args:
            key: The key of the group to be returned.

        Returns:
            The list corresponding to the supplied key.
        '''
        if key in self._dict:
            return self._dict[key]

        return []

    def __len__(self):
        '''Support for the len() built-in function.

        Returns:
            The number of Groupings (keys) in the lookup.'''
        return len(self._dict)

    def __contains__(self, key):
        '''Support for the 'in' membership operator.

        Args:
            key: The key for which membership will be tested.

        Returns:
            True if the Lookup contains a Grouping for the specified key,
            otherwise False.'''

        return key in self._dict

    def __repr__(self):
        '''Support for the repr() built-in function.

        Returns:
            The official string representation of the object.
        '''
        return 'Lookup({d})'.format(d=list(self._generate_repr_result()))

    def _generate_repr_result(self):
        for key in self._dict:
            for value in self._dict[key]:
                yield (key, value)
