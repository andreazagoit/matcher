import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const USER_FRAGMENT = gql`
  fragment UserFields on User {
    id
    username
    name
    email
    birthdate
    gender
    image
    createdAt
    updatedAt
    tags
    sexualOrientation
    heightCm
    relationshipIntent
    relationshipStyle
    hasChildren
    wantsChildren
    religion
    smoking
    drinking
    activityLevel
    jobTitle
    educationLevel
    schoolName
    languages
    ethnicity
    locationText
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_ME = gql`
  ${USER_FRAGMENT}
  query GetMe {
    me {
      ...UserFields
    }
  }
`;

export const GET_USER = gql`
  ${USER_FRAGMENT}
  query GetUser($username: String!) {
    user(username: $username) {
      ...UserFields
      userItems {
        id
        type
        promptKey
        content
        displayOrder
      }
    }
  }
`;

export const GET_USER_WITH_CARDS = gql`
  ${USER_FRAGMENT}
  query GetUserWithCards($username: String!) {
    user(username: $username) {
      ...UserFields
    }
  }
`;

export const CHECK_USERNAME = gql`
  query CheckUsername($username: String!) {
    checkUsername(username: $username)
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const UPDATE_USER = gql`
  ${USER_FRAGMENT}
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      ...UserFields
    }
  }
`;

export const GET_RECOMMENDED_TAGS = gql`
  query GetRecommendedTags($limit: Int) {
    me {
      id
      recommendedUserTags(limit: $limit)
    }
  }
`;

export const GET_RECOMMENDED_USERS = gql`
  query GetRecommendedUsers($limit: Int, $offset: Int) {
    me {
      id
      recommendedUserUsers(limit: $limit, offset: $offset) {
        id
        username
        name
        image
        birthdate
        gender
        userItems {
          id
          type
          content
          displayOrder
        }
      }
    }
  }
`;

export const UPDATE_MY_TAGS = gql`
  mutation UpdateMyTags($tags: [String!]!) {
    updateMyTags(tags: $tags) {
      id
      tags
    }
  }
`;

export const UPDATE_LOCATION = gql`
  mutation UpdateLocation($lat: Float!, $lon: Float!, $locationText: String) {
    updateLocation(lat: $lat, lon: $lon, locationText: $locationText) {
      id
      location {
        lat
        lon
      }
      locationUpdatedAt
    }
  }
`;

