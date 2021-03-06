import logging
import pymongo
import pandas as pds
import expiringdict
import utils

connection_url = 'mongodb+srv://movie:moviedata1050@movie.4icao.mongodb.net/<dbname>?retryWrites=true&w=majority'
client = pymongo.MongoClient(connection_url)
#logger = logging.Logger(__name__)
#utils.setup_logger(logger, 'db.log')     # database.log
RESULT_CACHE_EXPIRATION = 15             # seconds
   
    
def upsert_movie(df):
    """
    Update MongoDB database `movies` and collection `pop_movies` with the given `DataFrame`.
    """
    
    db = client.get_database("movies")
    collection = db.get_collection("pop_movies")
    db.pop_movies.remove({})
    update_count = 0
    for record in df.to_dict('records'):
        result = collection.replace_one(
            filter=record,    # locate the document if exists
            replacement=record,                         # latest document
            upsert=True)                                # update if exists, insert if not
#         if result.matched_count > 0:
#             update_count += 1
    #logger.info("rows={}, update={}, ".format(df.shape[0], update_count) +
                #"insert={}".format(df.shape[0]-update_count))


def fetch_all_movie():
    db = client.get_database("movies")
    collection = db.get_collection("pop_movies")
    ret = list(collection.find())
    return ret


_fetch_all_movies_as_df_cache = expiringdict.ExpiringDict(max_len=1,
                                                       max_age_seconds=RESULT_CACHE_EXPIRATION)


def fetch_all_movies_as_df(allow_cached=False):
    """Converts list of dicts returned by `fetch_all_movie` to DataFrame with ID removed
    Actual job is done in `_worker`. When `allow_cached`, attempt to retrieve timed cached from
    `_fetch_all_movies_as_df_cache`; ignore cache and call `_work` if cache expires or `allow_cached`
    is False.
    """
    def _work():
        data = fetch_all_movie()
        if len(data) == 0:
            return None
        df = pds.DataFrame.from_records(data)
        df.drop('_id', axis=1, inplace=True)
        return df

    if allow_cached:
        try:
            return _fetch_all_movies_as_df_cache['cache']
        except KeyError:
            pass
    ret = _work()
    _fetch_all_movies_as_df_cache['cache'] = ret
    return ret


if __name__ == '__main__':
    print(fetch_all_movies_as_df())  