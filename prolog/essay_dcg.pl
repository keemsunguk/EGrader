:- use_module(library(http/json)).
:- use_module(library(http/http_open)).
:- use_module(library(iostream)).
:- use_module(library(http/json_convert)).
:- use_module(library(lists)).


dcg_main(Idx, Defs) :-
        ky(Idx, Shallow),
        writeln(tokens(Shallow)),
    once(p_def(Defs, Shallow, [])).


p_def(Defs) -->
    [as_started + _, text_ref + T], labels_def(L), p_def(O_Defs),
    { O_Defs.put(specified_in, [T|L]) = Defs }.


p_def(Defs) -->
    {wide_op(Op)},
    [Op + _],
    p_def(Condition),
    { _{ }.put(Op, Condition) = Def}.

p_def(_{def_of: D, definition: Def}) --> [quoted_str + D, def + _], p_def(Def).
p_def(_{def_of: D, definition: Def, not_quoted: true}) --> [gap + D, def + _], p_def(Def).


