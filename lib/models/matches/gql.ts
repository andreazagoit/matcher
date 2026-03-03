import { gql } from "graphql-tag";

export const GET_DAILY_MATCHES = gql`
  query GetDailyMatches {
    me {
      id
      dailyMatches {
        score
        distanceKm
        user {
          id
          username
          name
          image
          gender
          birthdate
          userItems {
            id
            type
            content
            displayOrder
          }
        }
      }
    }
  }
`;
