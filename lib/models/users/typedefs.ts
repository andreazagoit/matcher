/**
 * GraphQL Schema for Users
 */

export const userTypeDefs = `#graphql
  """
  User — demographic, auth, and location data
  """
  type User {
    id: ID!
    username: String
    name: String!
    email: String!
    birthdate: String!
    gender: Gender
    image: String
    location: Location
    locationUpdatedAt: DateTime
    createdAt: DateTime!
    updatedAt: DateTime!
    interests: [UserInterest!]!
    profileItems: [ProfileItem!]!

    # Orientation & identity
    sexualOrientation: [String!]!
    heightCm: Int

    # Relational intent
    relationshipIntent: [String!]!
    relationshipStyle: RelationshipStyle
    hasChildren: HasChildren
    wantsChildren: WantsChildren

    # Lifestyle
    religion: Religion
    smoking: Smoking
    drinking: Drinking
    activityLevel: ActivityLevel

    # Identity & background
    jobTitle: String
    educationLevel: EducationLevel
    schoolName: String
    languages: [String!]!
    ethnicity: Ethnicity
  }

  type Location {
    lat: Float!
    lon: Float!
  }

  input CreateUserInput {
    name: String!
    email: String!
    birthdate: String!
    gender: Gender
  }

  input UpdateUserInput {
    username: String
    name: String
    email: String
    birthdate: String
    gender: Gender

    # Orientation & identity
    sexualOrientation: [String!]
    heightCm: Int

    # Relational intent
    relationshipIntent: [String!]
    relationshipStyle: RelationshipStyle
    hasChildren: HasChildren
    wantsChildren: WantsChildren

    # Lifestyle
    religion: Religion
    smoking: Smoking
    drinking: Drinking
    activityLevel: ActivityLevel

    # Identity & background
    jobTitle: String
    educationLevel: EducationLevel
    schoolName: String
    languages: [String!]
    ethnicity: Ethnicity
  }

  enum Gender {
    man
    woman
    non_binary
  }

  enum RelationshipStyle {
    monogamous
    ethical_non_monogamous
    open
    other
  }

  enum HasChildren {
    no
    yes
  }

  enum WantsChildren {
    yes
    no
    open
  }

  enum Religion {
    none
    christian
    muslim
    jewish
    buddhist
    hindu
    spiritual
    other
  }

  enum Smoking {
    never
    sometimes
    regularly
  }

  enum Drinking {
    never
    sometimes
    regularly
  }

  enum ActivityLevel {
    sedentary
    light
    moderate
    active
    very_active
  }

  enum EducationLevel {
    middle_school
    high_school
    bachelor
    master
    phd
    vocational
    other
  }

  enum Ethnicity {
    white_caucasian
    hispanic_latino
    black_african
    east_asian
    south_asian
    middle_eastern
    pacific_islander
    indigenous
    mixed
    other
  }

  extend type Query {
    user(username: String!): User
    users: [User!]!
    me: User
    checkUsername(username: String!): Boolean!
  }

  extend type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
    updateLocation(lat: Float!, lon: Float!): User!
  }
`;
