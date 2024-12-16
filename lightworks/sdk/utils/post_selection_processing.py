# Copyright 2024 Aegiq Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from types import FunctionType, NoneType

from ..state import State
from . import PostSelectionFunction
from .post_selection import PostSelectionType


def process_post_selection(
    post_selection: PostSelectionType | Callable[[State], bool] | None,
) -> PostSelectionType | None:
    """
    Takes a provided post-selection value and converts this into one of the
    PostSelection objects with a validate method.

    Args:

        post_selection (PostSelectionType | Callable | None) : The
            post-selection value to be processed. If a callable this will be
            converted into the PostSelectionFunction object, otherwise if None
            then the DefaultPostSelection object will be used.

    Returns:

        PostSelectionType | None : The created PostSelection object or None, if
            the original value wasn't one of these types.

    """
    if isinstance(post_selection, FunctionType):
        post_selection = PostSelectionFunction(post_selection)
    elif not isinstance(post_selection, PostSelectionType | NoneType):
        raise TypeError(
            "Provided post_select value should be a PostSelection object, "
            "function or None."
        )
    return post_selection
