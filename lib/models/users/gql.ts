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
    location
  }
`;

// ============================================
// QUERIES
// ============================================

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

export const UPDATE_LOCATION = gql`
  mutation UpdateLocation($lat: Float!, $lon: Float!, $location: String) {
    updateLocation(lat: $lat, lon: $lon, location: $location) {
      id
      coordinates {
        lat
        lon
      }
      location
      locationUpdatedAt
    }
  }
`;

export const GET_USER_RECOMMENDED_EVENTS = gql`
  query GetUserRecommendedEvents($limit: Int, $offset: Int) {
    me {
      id
      recommendedEvents(limit: $limit, offset: $offset) {
        id
        title
        description
        location
        coordinates { lat lon }
        startsAt
        endsAt
        attendeeCount
        categories
        spaceId
      }
    }
  }
`;

export const GET_RECOMMENDED_SPACES = gql`
  query GetRecommendedSpaces($limit: Int, $offset: Int) {
    me {
      id
      recommendedSpaces(limit: $limit, offset: $offset) {
        id
        name
        slug
        description
        cover
        categories
        visibility
        joinPolicy
        membersCount
        stripeAccountEnabled
        createdAt
      }
    }
  }
`;

export const GET_RECOMMENDED_USERS = gql`
  query GetRecommendedUsers($limit: Int, $offset: Int) {
    me {
      id
      recommendedUsers(limit: $limit, offset: $offset) {
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

export const GET_RECOMMENDED_CATEGORIES_WITH_EVENTS = gql`
  query GetRecommendedCategoriesWithEvents($limit: Int, $offset: Int) {
    me {
      id
      recommendedCategories(limit: $limit, offset: $offset) {
        id
        recommendedEvents(limit: 8) {
          id
          title
          description
          location
          startsAt
          endsAt
          attendeeCount
          categories
          spaceId
        }
      }
    }
  }
`;

