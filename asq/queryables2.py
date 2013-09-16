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
import itertools
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
    def __init__(self, name, iterable, query):
        self.dependencies = []  # Things who are dependent on me
        self.iterable = iterable
        self.query = query

    def add_dependency(self, collection):
        self.dependencies.append(collection)

    def insert(self, row):
        self.iterable.append(row)
        for dep in self.dependencies:
            dep.receive_dep_insert(self, row)

    def incr(self, row, col):
        self.iterable[row.id].__dict__[col] += 1

    def sum(self, row, col):
        self.iterable[row.id].__dict__[col] += row.__dict__[col]

    def receive_dep_insert(self, child_collection, dep_row):
        """ Receive an insert from a collection on which I depend.
        Apply it if necessary"""
        self.query.apply_new_row(self, dep_row)


class Queryable(object):
    '''Queries over iterables executed serially.

    Queryable objects are constructed from iterables.
    '''

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

    def __init__(self, iterable):
        '''Construct a Queryable from any iterable.

        Args:
            iterable: Any object supporting the iterator protocol.

        Raises:
            TypeError: if iterable does not support the iterator protocol.
        '''
        if not is_iterable(iterable):
            raise TypeError("Cannot construct Queryable from non-iterable {0}"
                .format(str(type(iterable))[7: -2]))

        self._iterable = iterable
        self.where = []

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

        return self._iter()

    def _iter(self):
        '''Return an unsorted iterator over the iterable.

        Useful in subclasses to obtain a raw iterator over the iterable where
        __iter__ has been overridden.
        '''
        return iter(self._iterable)

    def _create(self, iterable):
        '''Create a Queryable using the the supplied iterable.

        This method exists to allow it to be overridden by subclasses of
        Queryable.

        Args:
            iterable: An iterable.

        Returns:
            A Queryable constructed using the supplied iterable.

        Raises:
            TypeError: If the argument is not in fact iterable.
        '''
        return Queryable(iterable)

    def _create_ordered(self, iterable, direction, func):
        '''Create an ordered iterable using the supplied iterable.

        This method exists to allow it to be overridden by subclasses of
        Queryable.

        Args:
            iterable: The iterable sequence to be ordered.
            order: +1 for ascending, -1 for descending.
            func: The function to select the sorting key.
        '''
        return OrderedQueryable(iterable, direction, func)

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

        try:
            selector = make_selector(selector)
        except ValueError:
            raise TypeError("select() parameter selector={selector} cannot be"
                            "converted into a callable "
                            "selector".format(selector=repr(selector)))

        self.selector = selector

        if selector is identity:
            return self

        return self._create(imap(selector, self))


    def select_many(self, collection_selector=identity,
                    result_selector=identity):
        '''Projects each element of a sequence to an intermediate new sequence,
        flattens the resulting sequences into one sequence and optionally
        transforms the flattened sequence using a selector function.

        Note: This method uses deferred execution.

        Args:
            collection_selector: A unary function mapping each element of the
                source iterable into an intermediate sequence. The single
                argument of the collection_selector is the value of an element
                from the source sequence. The return value should be an
                iterable derived from that element value. The default
                collection_selector, which is the identity function, assumes
                that each element of the source sequence is itself iterable.

            result_selector: An optional unary function mapping the elements in
                the flattened intermediate sequence to corresponding elements
                of the result sequence. The single argument of the
                result_selector is the value of an element from the flattened
                intermediate sequence. The return value should be the
                corresponding value in the result sequence. The default
                result_selector is the identity function.

        Returns:
            A Queryable over a generated sequence whose elements are the result
            of applying the one-to-many collection_selector to each element of
            the source sequence, concatenating the results into an intermediate
            sequence, and then mapping each of those elements through the
            result_selector into the result sequence.

        Raises:
            ValueError: If this Queryable has been closed.
            TypeError: If either collection_selector or result_selector are not
                callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call select_many() on a closed "
                             "Queryable.")

        if not is_callable(collection_selector):
            raise TypeError("select_many() parameter projector={0} is not "
                            "callable".format(repr(collection_selector)))

        if not is_callable(result_selector):
            raise TypeError("select_many() parameter selector={selector} is "
                        " not callable".format(selector=repr(result_selector)))

        sequences = self.select(collection_selector)
        chained_sequence = itertools.chain.from_iterable(sequences)
        return self._create(chained_sequence).select(result_selector)

    def group_by(self, key_selector=identity,
                 element_selector=identity,
                 result_selector=lambda key, grouping: grouping):
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
                which simply returns the Grouping.

        Returns:
            A Queryable sequence of elements of the where each element
            represents a group.  If the default result_selector is relied upon
            this is a Grouping object.

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

        return self._create(self._generate_group_by_result(key_selector,
                            element_selector, result_selector))

    def _generate_group_by_result(self, key_selector, element_selector,
                                  result_selector):
        lookup = self.to_lookup(key_selector, element_selector)
        for grouping in lookup:
            yield result_selector(grouping.key, grouping)

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

        self.where = predicate

        return self._create(ifilter(predicate, self))

    def of_type(self, classinfo):
        '''Filters elements according to whether they are of a certain type.

        Note: This method uses deferred execution.

        Args:
            classinfo: If classinfo is neither a class object nor a type object
                it may be a tuple of class or type objects, or may recursively
                contain other such tuples (other sequence types are not
                accepted).

        Returns:
            A Queryable over those elements of the source sequence for which
            the predicate is True.

        Raises:
            ValueError: If the Queryable is closed.
            TypeError: If classinfo is not a class, type, or tuple of classes,
                types, and such tuples.
        '''
        if self.closed():
            raise ValueError("Attempt to call of_type() on a closed "
                             "Queryable.")

        if not is_type(classinfo):
            raise TypeError("of_type() parameter classinfo={0} is not a class "
                "object or a type objector a tuple of class or "
                "type objects.".format(classinfo))

        return self.where(lambda x: isinstance(x, classinfo))

    def order_by(self, key_selector=identity):
        '''Sorts by a key in ascending order.

        Introduces a primary sorting order to the sequence. Additional sort
        criteria should be specified by subsequent calls to then_by() and
        then_by_descending().  Calling order_by() or order_by_descending() on
        the results of a call to order_by() will introduce a new primary
        ordering which will override any already established ordering.

        This method performs a stable sort. The order of two elements with the
        same key will be preserved.

        Note: This method uses deferred execution.

        Args:
            key_selector: A unary function which extracts a key from each
                element using which the result will be ordered.

        Returns:
            An OrderedQueryable over the sorted elements.

        Raises:
            ValueError: If the Queryable is closed.
            TypeError: If the key_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call order_by() on a "
                             "closed Queryable.")

        if not is_callable(key_selector):
            raise TypeError("order_by() parameter key_selector={key_selector} "
                    "is not callable".format(key_selector=repr(key_selector)))

        self.order_by = key_selector

        return self._create_ordered(iter(self), -1, key_selector)

    def order_by_descending(self, key_selector=identity):
        '''Sorts by a key in descending order.

        Introduces a primary sorting order to the sequence. Additional sort
        criteria should be specified by subsequent calls to then_by() and
        then_by_descending().  Calling order_by() or order_by_descending() on
        the results of a call to order_by() will introduce a new primary
        ordering which will override any already established ordering.

        This method performs a stable sort. The order of two elements with the
        same key will be preserved.

        Note: This method uses deferred execution.

        Args:
            key_selector: A unary function which extracts a key from each
                element using which the result will be ordered.

        Returns:
            An OrderedQueryable over the sorted elements.

        Raises:
            ValueError: If the Queryable is closed.
            TypeError: If the key_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call order_by_descending() on a "
                             "closed Queryable.")

        if not is_callable(key_selector):
            raise TypeError("order_by_descending() parameter key_selector={0} "
                            "is not callable".format(repr(key_selector)))

        self.order_by = key_selector

        return self._create_ordered(iter(self), +1, key_selector)

    def take(self, count=1):
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

        return self._create(itertools.islice(self, count))

    def element_at(self, index):
        '''Return the element at ordinal index.

        Note: This method uses immediate execution.

        Args:
            index: The index of the element to be returned.

        Returns:
            The element at ordinal index in the source sequence.

        Raises:
            ValueError: If the Queryable is closed().
            ValueError: If index is out of range.
        '''
        if self.closed():
            raise ValueError("Attempt to call element_at() on a "
                             "closed Queryable.")

        if index < 0:
            raise OutOfRangeError("Attempt to use negative index.")

        # Attempt to use __getitem__
        try:
            return self._iterable[index]
        except IndexError:
            raise OutOfRangeError("Index out of range.")
        except TypeError:
            pass

        # Fall back to iterating
        for i, item in enumerate(self):
            if i == index:
                return item
        raise OutOfRangeError("element_at(index) out of range.")

    def count(self, predicate=None):
        '''Return the number of elements (which match an optional predicate).

        Note: This method uses immediate execution.

        Args:
            predicate: An optional unary predicate function used to identify
                elements which will be counted. The single positional argument
                of the function is the element value. The function should
                return True or False.

        Returns:
            The number of elements in the sequence if the predicate is None
            (the default), or if the predicate is supplied the number of
            elements for which the predicate evaluates to True.

        Raises:
            ValueError: If the Queryable is closed().
            TypeError: If predicate is neither None nor a callable.
        '''

        if self.closed():
            raise ValueError("Attempt to call element_at() on a "
                             "closed Queryable.")

        self.count = True
        self.count_predicate = predicate
        return self._count() if predicate is None else self._count_predicate(predicate)

    def _count(self):
        # Attempt to use len()
        try:
            return len(self._iterable)
        except TypeError:
            pass

        # Fall back to iterating
        index = -1

        for index, item in enumerate(self):
            pass

        return index + 1

    def _count_predicate(self, predicate):
        if not is_callable(predicate):
            raise TypeError("count() parameter predicate={0} is "
                            "not callable".format(repr(predicate)))

        return self.where(predicate).count()

    def min(self, selector=identity):
        '''Return the minimum value in a sequence.

        All of the source sequence will be consumed.

        Note: This method uses immediate execution.

        Args:
            selector: An optional single argument function which will be used
                to project the elements of the sequence. If omitted, the
                identity function is used.

        Returns:
            The minimum value of the projected sequence.

        Raises:
            ValueError: If the Queryable has been closed.
            ValueError: If the sequence is empty.
        '''
        if self.closed():
            raise ValueError("Attempt to call min() on a closed Queryable.")

        if not is_callable(selector):
            raise TypeError("min() parameter selector={0} is "
                            "not callable".format(repr(selector)))

        return min(self.select(selector))

    def max(self, selector=identity):
        '''Return the maximum value in a sequence.

        All of the source sequence will be consumed.

        Note: This method uses immediate execution.

        Args:
            selector: An optional single argument function which will be used
                to project the elements of the sequence. If omitted, the
                identity function is used.

        Returns:
            The maximum value of the projected sequence.

        Raises:
            ValueError: If the Queryable has been closed.
            ValueError: If the sequence is empty.
        '''

        if self.closed():
            raise ValueError("Attempt to call max() on a closed Queryable.")

        if not is_callable(selector):
            raise TypeError("max() parameter selector={0} is "
                            "not callable".format(repr(selector)))

        return max(self.select(selector))

    def sum(self, selector=identity):
        '''Return the arithmetic sum of the values in the sequence..

        All of the source sequence will be consumed.

        Note: This method uses immediate execution.

        Args:
            selector: An optional single argument function which will be used
                to project the elements of the sequence. If omitted, the
                identity function is used.

        Returns:
            The total value of the projected sequence, or zero for an empty
            sequence.

        Raises:
            ValueError: If the Queryable has been closed.
        '''

        if self.closed():
            raise ValueError("Attempt to call sum() on a closed Queryable.")

        if not is_callable(selector):
            raise TypeError("sum() parameter selector={0} is "
                            "not callable".format(repr(selector)))

        return sum(self.select(selector))

    def average(self, selector=identity):
        '''Return the arithmetic mean of the values in the sequence..

        All of the source sequence will be consumed.

        Note: This method uses immediate execution.

        Args:
            selector: An optional single argument function which will be used
                to project the elements of the sequence. If omitted, the
                identity function is used.

        Returns:
            The arithmetic mean value of the projected sequence.

        Raises:
            ValueError: If the Queryable has been closed.
            ValueError: I the source sequence is empty.
        '''
        if self.closed():
            raise ValueError("Attempt to call average() on a "
                             "closed Queryable.")

        if not is_callable(selector):
            raise TypeError("average() parameter selector={0} is "
                            "not callable".format(repr(selector)))

        total = 0
        count = 0
        for item in self.select(selector):
            total += item
            count += 1
        if count == 0:
            raise ValueError("Cannot compute average() of an empty sequence.")
        return total / count

    def default_if_empty(self, default):
        '''If the source sequence is empty return a single element sequence
        containing the supplied default value, otherwise return the source
        sequence unchanged.

        Note: This method uses deferred execution.

        Args:
            default: The element to be returned if the source sequence is empty.

        Returns:
            The source sequence, or if the source sequence is empty an sequence
            containing a single element with the supplied default value.

        Raises:
            ValueError: If the Queryable has been closed.
        '''

        if self.closed():
            raise ValueError("Attempt to call default_if_empty() on a "
                             "closed Queryable.")

        return self._create(self._generate_default_if_empty_result(default))

    def _generate_default_if_empty_result(self, default):
        # Try to get an element from the iterator, if we succeed, the sequence
        # is non-empty. We store the extracted value in a generator and chain
        # it to the tail of the sequence in order to recreate the original
        # sequence.
        try:
            items = iter(self)
            head = next(items)

            yield head

            for item in items:
                yield item

        except StopIteration:
            yield default

    def join(self, inner_iterable, outer_key_selector=identity,
             inner_key_selector=identity,
             result_selector=lambda outer, inner: (outer, inner)):
        '''Perform an inner join with a second sequence using selected keys.

        The order of elements from outer is maintained. For each of these the
        order of elements from inner is also preserved.

        Note: This method uses deferred execution.

        Args:
            inner_iterable: The sequence to join with the outer sequence.

            outer_key_selector: An optional unary function to extract keys from
                elements of the outer (source) sequence. The first positional
                argument of the function should accept outer elements and the
                result value should be the key. If omitted, the identity
                function is used.

            inner_key_selector: An optional  unary function to extract keys
                from elements of the inner_iterable. The first positional
                argument of the function should accept outer elements and the
                result value should be the key.  If omitted, the identity
                function is used.

            result_selector: An optional binary function to create a result
                element from two matching elements of the outer and inner. If
                omitted the result elements will be a 2-tuple pair of the
                matching outer and inner elements.

        Returns:
            A Queryable whose elements are the result of performing an inner-
            join on two sequences.

        Raises:
            ValueError: If the Queryable has been closed.
            TypeError: If the inner_iterable is not in fact iterable.
            TypeError: If the outer_key_selector is not callable.
            TypeError: If the inner_key_selector is not callable.
            TypeError: If the result_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call join() on a closed Queryable.")

        if not is_iterable(inner_iterable):
            raise TypeError("Cannot compute join() with inner_iterable of "
                   "non-iterable {0}".format(str(type(inner_iterable))[7: -1]))

        if not is_callable(outer_key_selector):
            raise TypeError("join() parameter outer_key_selector={0} is not "
                            "callable".format(repr(outer_key_selector)))

        if not is_callable(inner_key_selector):
            raise TypeError("join() parameter inner_key_selector={0} is not "
                            "callable".format(repr(inner_key_selector)))

        if not is_callable(result_selector):
            raise TypeError("join() parameter result_selector={0} is not "
                            "callable".format(repr(result_selector)))

        return self._create(self._generate_join_result(inner_iterable, outer_key_selector,
                                                       inner_key_selector, result_selector))

    def _generate_join_result(self, inner_iterable, outer_key_selector, inner_key_selector, result_selector):
        lookup = self._create(inner_iterable).to_lookup(inner_key_selector)
        result = self.select_many_with_correspondence(lambda outer_element: lookup[outer_key_selector(outer_element)],
                                                      result_selector)
        for item in result:
            yield item

    def group_join(self, inner_iterable, outer_key_selector=identity, inner_key_selector=identity,
             result_selector=lambda outer, grouping: grouping):
        '''Match elements of two sequences using keys and group the results.

        The group_join() query produces a hierarchical result, with all of the
        inner elements in the result grouped against the matching outer
        element.

        The order of elements from outer is maintained. For each of these the
        order of elements from inner is also preserved.

        Note: This method uses deferred execution.

        Args:
            inner_iterable: The sequence to join with the outer sequence.

            outer_key_selector: An optional unary function to extract keys from
                elements of the outer (source) sequence. The first positional
                argument of the function should accept outer elements and the
                result value should be the key. If omitted, the identity
                function is used.

            inner_key_selector: An optional  unary function to extract keys
                from elements of the inner_iterable. The first positional
                argument of the function should accept outer elements and the
                result value should be the key.  If omitted, the identity
                function is used.

            result_selector: An optional binary function to create a result
                element from an outer element and the Grouping of matching
                inner elements. The first positional argument is the outer
                elements and the second in the Grouping of inner elements
                which match the outer element according to the key selectors
                used. If omitted, the result elements will be the Groupings
                directly.

        Returns:
            A Queryable over a sequence with one element for each group in the
            result as returned by the result_selector. If the default result
            selector is used, the result is a sequence of Grouping objects.

        Raises:
            ValueError: If the Queryable has been closed.
            TypeError: If the inner_iterable is not in fact iterable.
            TypeError: If the outer_key_selector is not callable.
            TypeError: If the inner_key_selector is not callable.
            TypeError: If the result_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call group_join() on a closed Queryable.")

        if not is_iterable(inner_iterable):
            raise TypeError("Cannot compute group_join() with inner_iterable of non-iterable {type}".format(
                    type=str(type(inner_iterable))[7: -1]))

        if not is_callable(outer_key_selector):
            raise TypeError("group_join() parameter outer_key_selector={outer_key_selector} is not callable".format(
                    outer_key_selector=repr(outer_key_selector)))

        if not is_callable(inner_key_selector):
            raise TypeError("group_join() parameter inner_key_selector={inner_key_selector} is not callable".format(
                    inner_key_selector=repr(inner_key_selector)))

        if not is_callable(result_selector):
            raise TypeError("group_join() parameter result_selector={result_selector} is not callable".format(
                    result_selector=repr(result_selector)))

        return self._create(self._generate_group_join_result(inner_iterable, outer_key_selector,
                                                       inner_key_selector, result_selector))

    def _generate_group_join_result(self, inner_iterable, outer_key_selector, inner_key_selector, result_selector):
        lookup = self._create(inner_iterable).to_lookup(inner_key_selector)
        for outer_element in self:
            outer_key = outer_key_selector(outer_element)
            yield result_selector(outer_element, lookup[outer_key])


    def aggregate(self, reducer, seed=default, result_selector=identity):
        '''Apply a function over a sequence to produce a single result.

        Apply a binary function cumulatively to the elements of the source
        sequence so as to reduce the iterable to a single value.

        Note: This method uses immediate execution.

        Args:
            reducer: A binary function the first positional argument of which
                is an accumulated value and the second is the update value from
                the source sequence. The return value should be the new
                accumulated value after the update value has been incorporated.

            seed: An optional value used to initialise the accumulator before
                 iteration over the source sequence. If seed is omitted the
                 and the source sequence contains only one item, then that item
                 is returned.

            result_selector: An optional unary function applied to the final
                accumulator value to produce the result. If omitted, defaults
                to the identity function.

        Raises:
            ValueError: If called on an empty sequence with no seed value.
            TypeError: If reducer is not callable.
            TypeError: If result_selector is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call aggregate() on a "
                             "closed Queryable.")

        if not is_callable(reducer):
            raise TypeError("aggregate() parameter reducer={0} is "
                            "not callable".format(repr(reducer)))

        if not is_callable(result_selector):
            raise TypeError("aggregate() parameter result_selector={0} is "
                            "not callable".format(repr(result_selector)))

        if seed is default:
            try:
                return result_selector(fold(reducer, self))
            except TypeError as e:
                if 'empty sequence' in str(e):
                    raise ValueError("Cannot aggregate() empty sequence with "
                                     "no seed value")
        return result_selector(fold(reducer, self, seed))

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

    def to_lookup(self, key_selector=identity, value_selector=identity):
        '''Returns a Lookup object, using the provided selector to generate a
        key for each item.

        Note: This method uses immediate execution.
        '''
        if self.closed():
            raise ValueError("Attempt to call to_lookup() on a closed Queryable.")

        if not is_callable(key_selector):
            raise TypeError("to_lookup() parameter key_selector={key_selector} is not callable".format(
                    key_selector=repr(key_selector)))

        if not is_callable(value_selector):
            raise TypeError("to_lookup() parameter value_selector={value_selector} is not callable".format(
                    value_selector=repr(value_selector)))

        key_value_pairs = self.select(lambda item: (key_selector(item), value_selector(item)))
        lookup = Lookup(key_value_pairs)
        # Ideally we would close here
        #self.close()
        return lookup

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

        return str(separator).join(self.select(str))

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

    def log(self, logger=None, label=None, eager=False):
        '''
        Log query result consumption details to a logger.

        Args:
            logger: Any object which supports a debug() method which accepts a
                str, such as a Python standard library logger object from the
                logging module.  If logger is not provided or is None, this
                method has no logging side effects.

            label: An optional label which will be inserted into each line of
                logging output produced by this particular use of log

            eager: An optional boolean which controls how the query result will
                be consumed.  If True, the sequence will be consumed and logged
                in its entirety. If False (the default) the sequence will be
                evaluated and logged lazily as it consumed.

        Warning: Use of eager=True requires use of sufficient memory to
            hold the entire sequence which is obviously not possible with
            infinite sequences.  Use with care!

        Returns:
            A queryable over the unaltered source sequence.

        Raises:
            AttributeError: If logger does not support a debug() method.
            ValueError: If the Queryable has been closed.
        '''
        if self.closed():
            raise ValueError("Attempt to call log() on a closed Queryable.")

        if logger is None:
            return self

        if label is None:
            label = repr(self)

        if eager:
            return self._create(self._eager_log_result(logger, label))

        return self._create(self._generate_lazy_log_result(logger, label))

    def _eager_log_result(self, logger, label):
        seq1, seq2 = itertools.tee(self)
        logger.debug(label + " : BEGIN (EAGER)")

        for index, element in enumerate(seq1):
            logger.debug(label + ' : [' + str(index) + '] = ' + repr(element))

        logger.debug(label + " : END (EAGER)")
        return seq2

    def _generate_lazy_log_result(self, logger, label):

        logger.debug(label + " : BEGIN (DEFERRED)")

        for index, element in enumerate(self):
            logger.debug(label + ' : [' + str(index) + '] yields ' + repr(element))
            yield element

        logger.debug(label + " : END (DEFERRED)")

    def as_parallel(self, pool=None):
        '''Return a ParallelQueryable for parallel execution of queries.

        Warning: This feature should be considered experimental alpha quality.

        Args:
            pool: An optional multiprocessing pool which will provide execution
                resources for parellel processing.  If omitted, a pool will be
                created if necessary and managed internally.

        Returns:
            A ParallelQueryable on which all the standard query operators may
            be called.
        '''
        from .parallel_queryable import ParallelQueryable
        return ParallelQueryable(self, pool)

    # More operators

    def scan(self, func=operator.add):
        '''
        An inclusive prefix sum which returns the cumulative application of the
        supplied function up to an including the current element.

        Args:
             func: An optional binary function which is commutative - that is,
                 the order of the arguments is unimportant.  Defaults to a
                 summing operator.

        Returns:
            A Queryable such that the nth element is the sum of the first n
            elements of the source sequence.

        Raises:
            ValueError: If the Queryable has been closed.
            TypeError: If func is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call scan() on a "
                             "closed Queryable.")

        if not is_callable(func):
            raise TypeError("scan() parameter func={0} is "
                            "not callable".format(repr(func)))

        return self._create(self._generate_scan_result(func))

    def _generate_scan_result(self, func):

        i = iter(self)
        try:
            item = next(i)
            yield item
            accumulator = item
        except StopIteration:
            return

        for item in i:
            accumulator = func(accumulator, item)
            yield accumulator

    def pre_scan(self, func=operator.add, seed=0):
        '''
        An exclusive prefix sum which returns the cumulative application of the
        supplied function up to but excluding the current element.

        Args:
             func: An optional binary function which is commutative - that is,
                 the order of the arguments is unimportant.  Defaults to a
                 summing operator.

             seed: The first element of the prefix sum and therefore also the
                 first element of the returned sequence.

        Returns:
            A Queryable such that the nth element is the sum of the first n-1
            elements of the source sequence.

        Raises:
            ValueError: If the Queryable has been closed.
            TypeError: If func is not callable.
        '''
        if self.closed():
            raise ValueError("Attempt to call pre_scan() on a "
                             "closed Queryable.")

        if not is_callable(func):
            raise TypeError("pre_scan() parameter func={0} is "
                            "not callable".format(repr(func)))

        return self._create(self._generate_pre_scan_result(func, seed))

    def _generate_pre_scan_result(self, func, seed):

        accumulator = seed
        for item in self:
            yield accumulator
            accumulator = func(accumulator, item)


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


class Lookup(Queryable):
    '''A multi-valued dictionary.

    A Lookup represents a collection of keys, each one of which is mapped to
    one or more values. The keys in the Lookup are maintained in the order in
    which they were added. The values for each key are also maintained in
    order.

    Note: Lookup objects are immutable.

    All standard query operators may be used on a Lookup. When iterated or
    used as a Queryable the elements are returned as a sequence of Grouping
    objects.
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

        # Replace each list with a Grouping
        for key, value in iteritems(self._dict):
            grouping = Grouping(key, value)
            self._dict[key] = grouping
            
        super(Lookup, self).__init__(self._dict)

    def _iter(self):
        return itervalues(self._dict)

    def __getitem__(self, key):
        '''The sequence corresponding to a given key, or an empty sequence if
           there are no values corresponding to that key.

        Args:
            key: The key of the group to be returned.

        Returns:
            The Grouping corresponding to the supplied key.
        '''
        if key in self._dict:
            return self._dict[key]

        return Grouping(key, [])

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

    def apply_result_selector(self, selector=lambda key, sequence: sequence):
        return self._create(self._generate_apply_result_selector(selector))

    def _generate_apply_result_selector(self, selector):
        for grouping in self:
            yield selector(grouping.key, grouping)


class Grouping(Queryable):
    '''A collection of objects which share a common key.

    All standard query operators may be used on a Grouping.

    Note: It is not intended that clients should directly create Grouping
        objects. Instances of this class are retrieved from Lookup objects.
    '''

    def __init__(self, key, items):
        '''Create a Grouping with a given key and a collection of members.

        Args:
            key: The key corresponding to this Grouping

            items: An iterable collection of the members of the group.
        '''
        self._key = key
        sequence = list(items)
        super(Grouping, self).__init__(sequence)

    key = property(lambda self: self._key,
                   doc="The key common to all elements.")

    def __len__(self):
        '''The number of items in the Grouping.'''
        return self.count()

    def __eq__(self, rhs):
        '''Determine value equality with another grouping.

        Args:
           rhs: The object on the right-hand-side of the comparison must
                support a property called 'key' and be iterable.

        Returns:
            True if the keys and sequences are equal, otherwise False.
        '''
        return self.key == rhs.key and self.sequence_equal(rhs)

    def __ne__(self, rhs):
        '''Determine value inequality with another grouping.

        Args:
           rhs: The object on the right-hand-side of the comparison must
                support a property called 'key' and be iterable.

        Returns:
            True if the keys or sequences are not equal, otherwise False.
        '''
        return self.key != rhs.key or not self.sequence_equal(rhs)
    
    def __repr__(self):
        return 'Grouping(key={key}, items={items})'.format(key=repr(self._key),
                                                    items=repr(self.to_list()))

