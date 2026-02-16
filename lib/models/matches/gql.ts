import { gql } from "graphql-tag";

export const GET_FIND_MATCHES = gql`
  query GetFindMatches($maxDistance: Float = 50, $limit: Int) {
    findMatches(maxDistance: $maxDistance, limit: $limit) {
      user {
        id
        name
        givenName
        familyName
        image
        gender
        birthdate
      }
      similarity
      distance
      breakdown {
        psychological
        values
        interests
        behavioral
      }
    }
  }
`;

export const GET_PROFILE_STATUS = gql`
  query GetProfileStatus {
    profileStatus {
      hasAssessment
      hasProfile
      assessmentName
      completedAt
    }
  }
`;
