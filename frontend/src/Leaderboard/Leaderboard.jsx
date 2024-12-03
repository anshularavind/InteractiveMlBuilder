import React, {useEffect, useState} from "react";
import { useAuth0 } from "@auth0/auth0-react";
import "./Leaderboard.css";
import SearchBar from "./SearchBar";

function Leaderboard() {
    const {
        getAccessTokenSilently,
        loginWithRedirect,
        logout,
        user,
        isAuthenticated,
        isLoading,
    } = useAuth0();

    const [users, setUsers] = useState([]);
    const [searchTerm, setSearchTerm] = useState("");

    const getUsers = async () => {
        try {
            const token = await getAccessTokenSilently();
            console.log('Fetching users from backend...');
            const response = await fetch('http://localhost:4000/api/users', {
                method: 'GET',
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                    Accept: "application/json",
                },
                credentials: "include",
                mode: "cors",
            });

            if (!response.ok) {
                throw new Error(`Error fetching users: ${response.statusText}`);
            }
            const data = await response.json();
            if (data.users && Array.isArray(data.users)) {
                console.log('Users received:', data.users);
                return data.users;
            } else {
                throw new Error('Invalid data format for users.');
            }
        } catch (err) {
            console.error(err);
        }
    }

    useEffect(() => {
        getUsers()
        .then(users => {
            setUsers(users);
        })
        .catch(err => {
            console.error(err);
        });
    }, []);

    const handleSearch = (event) => {
        setSearchTerm(event.target.value.toLowerCase());
    };

    const filteredUsers = users.filter((user) =>
        user.username.toLowerCase().includes(searchTerm)
    );

    const displayUsers = filteredUsers.length === 0 ? users : filteredUsers;

    return (
        <div className="leaderboard">
        <h2>Leaderboard</h2>
        <SearchBar
                placeholder="Search by username"
                value={searchTerm}
                onChange={handleSearch}
                totalUsers={users.length}
                matchedUsers={filteredUsers.length}
        />
        <table>
            <thead>
            <tr>
                <th>Rank</th>
                <th>Username</th>
                <th>Models Trained</th>
            </tr>
            </thead>
            <tbody>
                {displayUsers.map((user, index) => (
                        <tr key={index}>
                            <td>{index + 1}</td>
                            <td>{user.username}</td>
                            <td>{user.num_models}</td>
                        </tr>
                ))}
            </tbody>
        </table>
        </div>
    );
}

export default Leaderboard;