import { MapPin, MessageSquareText } from "lucide-react";
import type { TweetRow } from "../lib/api";
import { formatNumber } from "../lib/api";

export function TweetTable({ tweets }: { tweets: TweetRow[] }) {
  return (
    <section className="panel tweet-panel">
      <div className="panel-heading">
        <span>Tweet Akışı</span>
        <h3>Öncelikli kayıtlar</h3>
      </div>
      <div className="tweet-list">
        {tweets.map((tweet) => (
          <article className="tweet-row" key={`${tweet.id}-${tweet.date}-${tweet.time}`}>
            <div className="tweet-meta">
              <span>
                <MessageSquareText size={15} /> {tweet.date} {tweet.time}
              </span>
              <span>
                <MapPin size={15} /> {[tweet.province, tweet.district, tweet.neighborhood].filter(Boolean).join(" / ") || "Konum yok"}
              </span>
              <b>U {formatNumber(tweet.urgency, 1)}</b>
            </div>
            <p>{tweet.text}</p>
            <div className="tweet-labels">
              {tweet.labels.map((label) => (
                <span key={label.id}>{label.name}</span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
