import React from "react";
import "./Leaderboard.css";

function SearchBar({ placeholder, value, onChange, totalUsers, matchedUsers }) {
    return (
        <div className="search-bar">
            <input
                type="text"
                placeholder={placeholder}
                value={value}
                onChange={onChange}
                className="search-input"
            />
            <p className="search-info">
                {matchedUsers === 0
                    ? `No matches found. Displaying all ${totalUsers} users.`
                    : `Displaying ${matchedUsers} matched ${
                          matchedUsers === 1 ? "user" : "users"
                      }.`}
            </p>
        </div>
    );
}

export default SearchBar;
