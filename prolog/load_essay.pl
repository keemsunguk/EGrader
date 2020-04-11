:- use_module(library(http/json)).
:- use_module(library(http/http_open)).
:- dynamic(known/2).
% open_notify_url("http://api.open-notify.org/iss-now.json").
open_notify_url("http://ec2-18-223-3-229.us-east-2.compute.amazonaws.com:8001/essay_details?recno=32636").

iss_data(Data) :-
    open_notify_url(URL),
    setup_call_cleanup(
        http_open(URL, In, [request_header('Accept'='application/json')]),
        json_read_dict(In, Data),
        close(In)
    ).

cached_iss_data(Data) :-
    known(data, Data) ;
    iss_data(Data),
    assert(known(data, Data)).

% iss_location(Data, Lat, Long) :-
%    Position = Data.get(iss_position),
%    Lat = Position.get(latitude),
%    Long = Position.get(longitude).

get_dep(Data, Cnst) :-
    Cnst is Data.get(constituency).
    Id