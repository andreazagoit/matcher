import { gql } from "graphql-tag";

export const GET_FIND_MATCHES = gql`
  query GetFindMatches($maxDistance: Float! = 50, $limit: Int, $offset: Int, $gender: [String!], $minAge: Int, $maxAge: Int) {
    findMatches(maxDistance: $maxDistance, limit: $limit, offset: $offset, gender: $gender, minAge: $minAge, maxAge: $maxAge) {
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
      score
      distanceKm
      sharedTags
      sharedSpaceIds
      sharedEventIds
    }
  }
`;

export const GET_PROFILE_STATUS = gql`
  query GetProfileStatus {
    profileStatus {
      hasProfile
      updatedAt
    }
  }
`;
